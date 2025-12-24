import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_mixer import DVDWMMixer
from modules.world_models import REGISTRY as wm_REGISTRY
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class DVDRssmLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # 1. 初始化 Mixer
        if args.mixer == "dvd_wm":
            self.mixer = DVDWMMixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised for DVDWMLearner")
        self.target_mixer = copy.deepcopy(self.mixer)

        # 2. 初始化 World Model (全局模式)
        # Input = Global State + Joint Actions (All Agents)
        self.state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(args.state_shape[0])
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        # 输入维度: State + (N_Agents * N_Actions)
        wm_in_shape = self.state_dim + (self.n_agents * self.n_actions)
        
        self.world_model = wm_REGISTRY[args.world_model_type](wm_in_shape, args)

        # 3. 定义优化器
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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # --- Forward Pass: RL Agent ---
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- Forward Pass: Target Network ---
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # [注意] 这里先保留完整序列 0..T，后面在计算 Target 时再切片 [1:]
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            # Double Q 选择动作
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # ==================== World Rollout & Latent Extraction ====================
        bs = batch.batch_size
        
        # 1. 构建输入: Global State + Joint Actions
        joint_actions = batch["actions_onehot"].reshape(bs, batch.max_seq_length, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1) # [B, T, State + N*A]
        
        # 初始化 Hidden State
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        z_history = []          # 用于存储传给 Mixer 的 z [0...T]
        forward_outputs = []    # 用于存储 RSSM 的中间输出以便计算 Loss

        # [Phase 1]: 主循环 (处理 t=0 到 T-1)
        # 这一步产生 h_1...h_T 和 z_0...z_{T-1}
        for t in range(batch.max_seq_length):
            # 1. WM Step (Forward)
            # wm_hidden 在输入时是 h_t, 输出时更新为 h_{t+1}
            wm_hidden, dist_params = self.world_model.forward_step(wm_inputs[:, t], wm_hidden)
            
            # 2. 收集输出供后续计算 Loss
            forward_outputs.append((wm_hidden, dist_params))

            # 3. Sample Z for Mixer 
            # 这里的 sample_latents 已经在 rssm_model.py 中修复为使用 [h_t, z_t]
            z_feature = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_feature)

        # [Phase 2]: 补全最后一步 (处理 t=T) [Critical Fix!]
        # 此时循环结束，wm_hidden 是 h_T
        # 我们需要计算 z_T (对应 state[:, -1]) 用于 Target Network 的最后一步输入
        # 因为没有 a_T，无法更新 RNN 到 h_{T+1}，但可以推断后验 z_T
        
        last_state = batch["state"][:, -1] # s_T
        # 调用 RSSM 的 infer_posterior 辅助函数 (需确保 rssm_model.py 已更新)
        z_T_params = self.world_model.infer_posterior(wm_hidden, last_state) 
        
        # 手动拼接 [h_T, z_T]
        # 注意保持维度一致: [1, B, Latent]
        z_T_feature = th.cat([wm_hidden, z_T_params['z']], dim=-1).unsqueeze(0)
        z_history.append(z_T_feature)

        # ==================== World Model Loss Calculation ====================
        # 依然只计算前 T-1 步的预测 Loss (预测 1...T)
        
        seq_len = batch.max_seq_length
        if seq_len > 1:
            valid_outputs = forward_outputs[:-1] 
            target_states = batch["state"][:, 1:] # s_1 ... s_T
            target_rewards = batch["reward"][:, :-1] # r_0 ... r_{T-1}

            recon_loss, kl_loss, reward_loss = self.world_model.compute_loss(
                valid_outputs, target_states, target_rewards
            )
            
            mask_squeeze = mask.squeeze(-1) # [B, T-1]
            
            wm_loss = (recon_loss * mask_squeeze).sum() / mask_squeeze.sum() + \
                      self.args.wm_kl_beta * (kl_loss * mask_squeeze).sum() / mask_squeeze.sum() + \
                      (reward_loss * mask_squeeze).sum() / mask_squeeze.sum()
        else:
            wm_loss = th.tensor(0.0).to(self.device)

        # ==================== Mixing & RL Loss ====================
        
        # Stack Z: [T+1, D, B, L] -> [D, B, T+1, L]
        # 现在 z_tensor 的长度是 T+1 (0...T)
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach()
        
        # --- Z-Warmup (Trick) ---
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False):
            if t_env < self.args.z_warmup_steps:
                warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        
        z_tensor_mixer = z_tensor * warmup_coef

        # [Final Correct Alignment]
        # 目标：对齐到 T-1 (即 0..T-2 索引，共 T-1 个时间步)
        # 假设 max_seq_length=T. 
        # chosen_action_qvals 长度为 T-1 (0..T-2).
        # z_tensor 长度为 T+1 (0..T).
        
        # 1. Prepare Eval inputs
        # z_eval 需要 0..T-2.
        # 切片 [:-2] 正好去掉最后两个 (T-1 和 T)，剩下 0..T-2.
        z_eval = z_tensor_mixer[:, :, :-2]
        
        # Mixer Forward (Current Q)
        # chosen_action_qvals 和 batch["state"][:, :-1] 已经是 T-1 长度，直接使用，无需再切片
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)
        
        with th.no_grad():
            # 2. Prepare Target inputs
            # z_target 需要 1..T-1 (对应 s_1..s_{T-1}).
            # 切片 [1:-1] 去掉头(0)和尾(T)，剩下 1..T-1.
            z_target = z_tensor_mixer[:, :, 1:-1]
            
            # Target Qs (from target_mac_out [0..T-1]).
            # 我们需要 1..T-1. 切片 [1:].
            target_max_qvals_next = target_max_qvals[:, 1:]
            
            # Target State [1..T-1]. 切片 batch["state"][:, 1:]
            target_state = batch["state"][:, 1:]
            
            # Target Mixer Input
            target_max_qvals = self.target_mixer(target_max_qvals_next, target_state, z_target)
            
            if getattr(self.args, 'q_lambda', False):
                 pass 
            else:
                # 注意：这里不需要再对 rewards/terminated/mask 进行额外切片
                # 它们原本就是 [:, :-1]，长度为 T-1，正好与 target_max_qvals 匹配
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # 3. Calculate Loss
        # 现在两者应该都是 T-1 长度 (例如 70)
        td_error = (chosen_action_qvals - targets.detach())
        loss_td = (td_error ** 2 * mask.expand_as(td_error)).sum() / mask.sum()
        
        # --- Optimization ---
        total_loss = loss_td + self.args.wm_loss_weight * wm_loss

        self.optimiser.zero_grad()
        self.wm_optimiser.zero_grad()
        
        total_loss.backward()
        
        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        th.nn.utils.clip_grad_norm_(self.wm_params, self.args.grad_norm_clip)
        
        self.optimiser.step()
        self.wm_optimiser.step()

        # --- Updates & Logging ---
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_wm", wm_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def evaluate_world_model(self, batch, t_env):
        # 1. 数据准备
        if batch.device != self.args.device:
            batch.to(self.args.device)
            
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        # 构建输入: Global State + Joint Actions
        joint_actions = batch["actions_onehot"].reshape(bs, max_t, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1)
        
        target_states = batch["state"][:, 1:] 
        mask = batch["filled"][:, :-1].float().squeeze(-1)

        # 2. 切换评估模式
        self.world_model.eval()
        pred_states_list = []
        
        with th.no_grad():
            wm_hidden = self.world_model.init_hidden(bs, self.device)
            
            for t in range(max_t - 1):
                # 预测 t+1
                # predict 内部已适配 [h_t, z_t] 逻辑
                pred_next_state, wm_hidden = self.world_model.predict(
                    wm_inputs[:, t], wm_hidden, use_mean=True
                )
                pred_states_list.append(pred_next_state)
            
            pred_states = th.stack(pred_states_list, dim=1)

            # 3. 计算指标
            errors = (pred_states - target_states) ** 2
            mse_per_step = errors.mean(dim=-1) 
            masked_mse = (mse_per_step * mask).sum() / mask.sum()
            
            # R^2 Score
            mask_bool = mask.bool()
            y_true_flat = target_states[mask_bool.unsqueeze(-1).expand_as(target_states)].reshape(-1, self.state_dim)
            y_pred_flat = pred_states[mask_bool.unsqueeze(-1).expand_as(pred_states)].reshape(-1, self.state_dim)
            
            ss_res = ((y_true_flat - y_pred_flat) ** 2).sum()
            y_mean = y_true_flat.mean(dim=0)
            ss_tot = ((y_true_flat - y_mean) ** 2).sum()
            
            if ss_tot.item() == 0:
                r2_score = 0.0
            else:
                r2_score = 1 - (ss_res / ss_tot)

            # 4. 打印与日志记录
            log_prefix = "test_wm_"
            self.logger.log_stat(log_prefix + "mse", masked_mse.item(), t_env)
            self.logger.log_stat(log_prefix + "r2", r2_score.item(), t_env)
            
            print(f"\n[Test Evaluation] World Model Performance at t_env={t_env}:")
            print(f"  >>> MSE: {masked_mse.item():.6f}")
            print(f"  >>> R^2: {r2_score.item():.6f}")
            print("-" * 50)

        self.world_model.train()

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