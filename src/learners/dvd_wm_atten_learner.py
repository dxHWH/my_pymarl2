import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_mixer import DVDWMMixer
from modules.world_models import REGISTRY as wm_REGISTRY
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class DVDWMAttenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # 1. 初始化 Mixer (支持 HyperAttention)
        if args.mixer == "dvd_wm":
            self.mixer = DVDWMMixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised for DVDWMAttenLearner")
        self.target_mixer = copy.deepcopy(self.mixer)

        # 2. 初始化 World Model (支持 EntityAttention)
        self.state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(args.state_shape[0])
        # 注意: 如果 WM 内部已经改为显式分离 State 和 Action，这里的 wm_in_shape 可能只作为参考
        wm_in_shape = self.state_dim + (args.n_agents * args.n_actions)
        self.world_model = wm_REGISTRY[args.world_model_type](wm_in_shape, args)

        # 3. 优化器配置
        # 将 Mixer 和 Agent 参数打包
        self.rl_params = list(mac.parameters()) + list(self.mixer.parameters())
        self.wm_params = list(self.world_model.parameters())

        if args.optimizer == 'adam':
            self.optimiser = Adam(params=self.rl_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.wm_optimiser = Adam(params=self.wm_params, lr=args.wm_lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.rl_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.wm_optimiser = RMSprop(params=self.wm_params, lr=args.wm_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        # 4. 对齐系数 (建议在 yaml 中配置 align_beta)
        self.align_beta = getattr(args, "align_beta", 0.01) 

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # --- 基础数据准备 ---
        max_seq_length = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        bs = batch.batch_size

        # =============================================================
        # 1. RL Agent Forward (计算 Q 值)
        # =============================================================
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1) # [B, T, N, A]
        
        # Chosen Action Q-Values
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) # [B, T-1, N]

        # Target Network Forward
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(bs)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            # Double Q-Learning Logic
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # =============================================================
        # 2. World Model Rollout (提取物理注意力)
        # =============================================================
        state_seq = batch["state"] # [B, T, S]
        actions_onehot_seq = batch["actions_onehot"] # [B, T, N, A]
        
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        recon_losses, kl_losses, reward_losses = [], [], []
        z_history = []
        wm_weights_history = [] # 存储 WM 的物理注意力权重 (Teacher)

        # 循环 Rollout
        for t in range(max_seq_length):
            # [关键]: 传入 actions 以便 WM 计算 EntityAttention
            # 假设 WM.forward_step 返回 (hidden, dist, attn_weights)
            wm_hidden, dist_params, wm_attn_w = self.world_model.forward_step(
                state_seq[:, t], wm_hidden, actions_onehot_seq[:, t]
            )
            
            # wm_attn_w: [B, N] (Sum=1, 已经 Softmax)
            wm_weights_history.append(wm_attn_w)
            
            # 采样 Latent Z (用于去混淆)
            z_samples = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_samples)
            
            # 计算 WM Loss
            if t < max_seq_length - 1:
                target_state = batch["state"][:, t+1]
                target_reward = batch["reward"][:, t]
                r_l, k_l, rew_l = self.world_model.compute_loss(
                    z_samples, wm_hidden, target_state, dist_params, target_reward
                )
                recon_losses.append(r_l); kl_losses.append(k_l); reward_losses.append(rew_l)

        # 聚合 WM Loss
        recon_stack = th.stack(recon_losses, dim=1)
        kl_stack = th.stack(kl_losses, dim=1)
        reward_stack = th.stack(reward_losses, dim=1)
        mask_squeeze = mask.squeeze(-1)
        
        # 维度截断保护
        min_len = min(recon_stack.shape[1], mask_squeeze.shape[1])
        wm_loss = (recon_stack[:, :min_len] * mask_squeeze[:, :min_len]).sum() / mask_squeeze[:, :min_len].sum() + \
                  self.args.wm_kl_beta * (kl_stack[:, :min_len] * mask_squeeze[:, :min_len]).sum() / mask_squeeze[:, :min_len].sum() + \
                  (reward_stack[:, :min_len] * mask_squeeze[:, :min_len]).sum() / mask_squeeze[:, :min_len].sum()

        # =============================================================
        # 3. Mixer Forward & 认知对齐 (提取价值注意力)
        # =============================================================
        # 准备 Z Tensor
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach() # [D, B, T, L]
        
        # Z-Warmup
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False) and t_env < self.args.z_warmup_steps:
             warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        z_tensor_mixer = z_tensor * warmup_coef
        
        # Mixer Input
        z_eval = z_tensor_mixer[:, :, :-1] # [D, B, T-1, L]
        
        # --- Mixer Forward ---
        # 注意：Mixer 内部计算 Attention 并保存在 hyper_attention.last_attn_weights
        mix_out = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)

        # --- DASA: Dual Attention Structural Alignment ---
        
        # 1. 准备 Teacher 分布 (WM Attention)
        # [B, T, N] -> 取前 T-1 步 (与 Mixer 对齐)
        # 这里的 weights 已经是概率分布 (Sum=1)
        wm_attn_seq = th.stack(wm_weights_history, dim=1)[:, :-1].detach() # [B, T-1, N]
        
        # 2. 准备 Student 分布 (Mixer Attention)
        # 从 Mixer 中提取 raw weights (Tanh 输出, 维度可能是 [D*B*(T-1), N, 1])
        # 假设 Mixer 内部做了 flatten，我们需要还原维度
        mixer_raw_weights = self.mixer.hyper_attention.last_attn_weights 
        
        # 还原维度: [D*B*(T-1), N, 1] -> [D, B, T-1, N]
        # 注意: 如果 D=1，直接 reshape
        d_samples = self.args.dvd_samples
        mixer_attn_reshaped = mixer_raw_weights.view(d_samples, bs, max_seq_length-1, self.args.n_agents)
        
        # 对 D 维度取平均 (Average over latent samples)
        mixer_attn_mean = mixer_attn_reshaped.mean(dim=0) # [B, T-1, N]
        
        # Softmax 归一化: 将 Mixer 的"关注强度"转化为"概率分布"
        # 即使 Tanh 输出有负数，abs() 后代表关注程度，softmax 后变成分布
        mixer_dist = F.softmax(th.abs(mixer_attn_mean), dim=-1) # [B, T-1, N] (Student)

        # 3. 计算 KL 散度 Loss
        # KL(Teacher || Student) = sum(P * (log P - log Q))
        # PyTorch F.kl_div(input, target) 中: input 是 log_prob (Student), target 是 prob (Teacher)
        
        # Student Log-Prob
        mixer_log_prob = torch.log(mixer_dist + 1e-10).view(-1, self.args.n_agents) # [B*(T-1), N]
        # Teacher Prob
        wm_prob = wm_attn_seq.reshape(-1, self.args.n_agents) # [B*(T-1), N]
        
        # 计算每个样本的 KL
        align_loss_raw = F.kl_div(mixer_log_prob, wm_prob, reduction='none').sum(dim=-1) # [B*(T-1)]
        
        # Apply Mask (只对非填充步计算 Loss)
        mask_flat = mask.reshape(-1)
        align_loss = (align_loss_raw * mask_flat).sum() / mask_flat.sum()

        # =============================================================
        # 4. 优化更新
        # =============================================================
        
        # 计算 TD Targets
        with th.no_grad():
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], z_tensor_mixer)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # TD Error
        td_error = (mix_out - targets.detach())
        if td_error.shape[1] != mask.shape[1]:
             min_len = min(td_error.shape[1], mask.shape[1])
             td_error = td_error[:, :min_len]
             mask = mask[:, :min_len]
        
        loss_td = (td_error ** 2 * mask.expand_as(td_error)).sum() / mask.sum()
        
        # Total Loss
        # Story: 强化学习目标 + 世界模型理解 + 认知结构对齐
        total_loss = loss_td + \
                     self.args.wm_loss_weight * wm_loss + \
                     self.align_beta * align_loss

        self.optimiser.zero_grad()
        self.wm_optimiser.zero_grad()
        total_loss.backward()
        
        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        th.nn.utils.clip_grad_norm_(self.wm_params, self.args.grad_norm_clip)
        
        self.optimiser.step()
        self.wm_optimiser.step()

        # Update Targets
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_wm", wm_loss.item(), t_env)
            self.logger.log_stat("loss_align", align_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.world_model.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.wm_optimiser.state_dict(), "{}/wm_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # 简单的 try-except 以防旧模型没有 wm 优化器
        try:
            self.wm_optimiser.load_state_dict(th.load("{}/wm_opt.th".format(path), map_location=lambda storage, loc: storage))
        except:
            pass