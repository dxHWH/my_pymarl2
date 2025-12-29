import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_causal_mixer import DVDWMCausalMixer
from modules.world_models import REGISTRY as wm_REGISTRY
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class DVDWMCausalLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # 1. Initialize Causal Mixer
        if args.mixer == "dvd_wm_causal":
            self.mixer = DVDWMCausalMixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised for DVDWMCausalLearner")
        self.target_mixer = copy.deepcopy(self.mixer)

        # 2. Initialize Standard World Model (vae_rnn)
        self.state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(args.state_shape[0])
        wm_in_shape = self.state_dim + (args.n_agents * args.n_actions)
        self.world_model = wm_REGISTRY[args.world_model_type](wm_in_shape, args)

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
        
        self.causal_beta = getattr(args, "causal_beta", 0.01)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # 获取 Batch 配置
        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # =================================================================
        # 1. RL Agent Forward (Full Batch)
        # =================================================================
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        
        # 提取当前动作对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Target MAC Forward
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(bs)
            for t in range(max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # =================================================================
        # 2. World Model & Causal Inference
        # =================================================================
        joint_actions = batch["actions_onehot"].reshape(bs, max_seq_length, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1)
        
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        recon_losses, kl_losses, reward_losses = [], [], []
        z_history = []
        causal_weights_history = [] 

        # 因果推断参数
        reward_sensitivity = 10.0  # 奖励差异的权重系数
        
        for t in range(max_seq_length):
            # [关键]: 在 Update 前缓存当前时刻 hidden (h_t)
            # 反事实推理的基础是: 给定当前的 h_t, 如果动作变了, h_{t+1} 会怎么变?
            current_hidden = wm_hidden.detach().clone()
            
            # --- A. Factual Pass (事实路径) ---
            input_t = wm_inputs[:, t]
            # wm_hidden 更新为 h_{t+1}
            wm_hidden, dist_params = self.world_model.forward_step(input_t, wm_hidden)
            
            # --- B. Causal Perturbation Pass (反事实路径) ---
            if t < max_seq_length - 1:
                with th.no_grad():
                    n_agents = self.args.n_agents
                    
                    # 1. 数据准备与扩展
                    # State [B, S] -> [B*N, S]
                    state_t = batch["state"][:, t]
                    state_repeat = state_t.repeat_interleave(n_agents, dim=0)
                    
                    # Hidden [B, H] -> [B*N, H] (使用缓存的 h_t)
                    hidden_repeat = current_hidden.repeat_interleave(n_agents, dim=0)
                    
                    # Action [B, N, A]
                    action_t = batch["actions_onehot"][:, t]

                    # 2. 构造反事实动作 (Counterfactual Actions)
                    # 目标: 将对角线上的 Agent 动作替换为 Action 1 (Stop)
                    
                    # (a) 扩展原始动作 [B, 1, N, A] -> [B, N, N, A]
                    action_exp = action_t.unsqueeze(1).expand(-1, n_agents, -1, -1).clone()
                    
                    # (b) 构造掩码
                    # mask_mat (保留项): 对角线为 0, 其他为 1
                    mask_mat = 1.0 - th.eye(n_agents, device=self.device).view(1, n_agents, n_agents, 1)
                    # inv_mask_mat (替换项): 对角线为 1, 其他为 0
                    inv_mask_mat = 1.0 - mask_mat
                    
                    # (c) 构造 Stop 动作向量 [1, 1, 1, A]
                    # Action 1 是 Stop, 对应的 One-hot 是 [0, 1, 0, 0, ...]
                    # 注意: 确保 n_actions >= 2
                    stop_action = th.zeros(1, 1, 1, self.args.n_actions, device=self.device)
                    if self.args.n_actions > 1:
                        stop_action[..., 1] = 1.0 
                    
                    # (d) 融合: 原始动作 * Mask + Stop动作 * InvMask
                    # 结果: 第 i 个平行宇宙中, Agent i 执行 Stop, 其他人保持原样
                    action_cf_4d = (action_exp * mask_mat) + (stop_action * inv_mask_mat)
                    
                    # Flatten -> [B * N, N * A]
                    action_cf_flat = action_cf_4d.view(bs * n_agents, -1)
                    
                    # 3. 构造 WM 输入
                    pert_input = th.cat([state_repeat, action_cf_flat], dim=-1)
                    
                    # 4. Counterfactual Forward (P(z | h_t, a_perturbed))
                    # 得到反事实的 h'_{t+1} 和分布参数
                    cf_hidden_next, cf_dist_params = self.world_model.forward_step(pert_input, hidden_repeat)
                    
                    # 5. 计算混合因果效应 (Hybrid Causal Effect)
                    
                    # --- Part 1: Physical Effect (KL Divergence) ---
                    # 准备 Factual 参数 (扩展以匹配维度)
                    mu_fac = dist_params[0].detach().unsqueeze(1).expand(-1, n_agents, -1)
                    logvar_fac = dist_params[1].detach().unsqueeze(1).expand(-1, n_agents, -1)
                    var_fac = th.exp(logvar_fac)
                    
                    # 准备 Counterfactual 参数 (Reshape)
                    mu_cf = cf_dist_params[0].view(bs, n_agents, -1)
                    logvar_cf = cf_dist_params[1].view(bs, n_agents, -1)
                    var_cf = th.exp(logvar_cf)
                    
                    # KL(P || Q)
                    term1 = var_fac / (var_cf + 1e-8)
                    term2 = (mu_fac - mu_cf).pow(2) / (var_cf + 1e-8)
                    term3 = logvar_cf - logvar_fac
                    kl_div = 0.5 * th.sum(term1 + term2 + term3 - 1, dim=-1) # [B, N]
                    
                    # --- Part 2: Value Effect (Reward Difference) ---
                    # 利用 Reward Head 预测奖励
                    
                    # Factual Reward: Head(z_fac, h_{t+1})
                    # 使用均值 z 预测 (deterministic)
                    z_fac = dist_params[0].detach() 
                    # 注意: 这里使用更新后的 wm_hidden (h_{t+1})
                    fac_input_reward = th.cat([z_fac, wm_hidden.detach()], dim=-1)
                    r_fac = self.world_model.reward_head(fac_input_reward) # [B, 1]
                    r_fac_exp = r_fac.unsqueeze(1).expand(-1, n_agents, -1) # [B, N, 1]
                    
                    # Counterfactual Reward: Head(z_cf, h'_{t+1})
                    z_cf = cf_dist_params[0] 
                    # 注意: 这里使用反事实的 next hidden
                    cf_input_reward = th.cat([z_cf, cf_hidden_next], dim=-1) 
                    r_cf = self.world_model.reward_head(cf_input_reward) # [B*N, 1]
                    r_cf_view = r_cf.view(bs, n_agents, -1) # [B, N, 1]
                    
                    # Reward Difference (MSE)
                    # "如果我变成Stop, 预测奖励会变多少?"
                    r_diff = (r_fac_exp - r_cf_view).pow(2).sum(dim=-1) # [B, N]
                    
                    # --- Combine Effects ---
                    total_effect = kl_div + reward_sensitivity * r_diff
                    
                    causal_weights = F.softmax(total_effect, dim=-1)
                    causal_weights_history.append(causal_weights)

            # --- C. WM Loss Calculation (Standard) ---
            z_samples = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_samples)
            
            if t < max_seq_length - 1:
                target_state = batch["state"][:, t+1]
                target_reward = batch["reward"][:, t]
                # 计算重建损失和奖励预测损失
                r_l, k_l, rew_l = self.world_model.compute_loss(
                    z_samples, wm_hidden, target_state, dist_params, target_reward
                )
                recon_losses.append(r_l); kl_losses.append(k_l); reward_losses.append(rew_l)

        # 聚合 WM Loss
        mask_squeeze = mask.squeeze(-1)
        min_len = min(len(recon_losses), mask_squeeze.shape[1])
        
        recon_stack = th.stack(recon_losses, dim=1)[:, :min_len]
        kl_stack = th.stack(kl_losses, dim=1)[:, :min_len]
        reward_stack = th.stack(reward_losses, dim=1)[:, :min_len]
        mask_wm = mask_squeeze[:, :min_len]
        
        wm_loss = (recon_stack * mask_wm).sum() / mask_wm.sum() + \
                  self.args.wm_kl_beta * (kl_stack * mask_wm).sum() / mask_wm.sum() + \
                  (reward_stack * mask_wm).sum() / mask_wm.sum()

        # =================================================================
        # 3. Mixer Forward & Alignment
        # =================================================================
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach() # [D, B, T, L]
        
        # Z Warmup
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False) and t_env < self.args.z_warmup_steps:
             warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        z_tensor_mixer = z_tensor * warmup_coef
        z_eval = z_tensor_mixer[:, :, :-1]
        
        # Mixer Forward
        mix_out = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)

        # Alignment Loss
        if len(causal_weights_history) > 0:
            target_probs = th.stack(causal_weights_history, dim=1) # [B, T-1, N]
            target_probs = th.clamp(target_probs, min=1e-6, max=1.0-1e-6)
            
            # 获取 Mixer 的 Attention 权重
            mixer_raw = self.mixer.hyper_attention.last_attn_weights # [D*B*T, N, 1] or similar
            d_samples = self.args.dvd_samples
            # Reshape & Mean: [D, B, T, N] -> [B, T, N]
            mixer_attn = mixer_raw.view(d_samples, bs, -1, self.args.n_agents).mean(dim=0)
            
            # 截断以匹配长度
            valid_T = target_probs.shape[1]
            mixer_attn = mixer_attn[:, :valid_T, :]
            
            mixer_log_prob = F.log_softmax(th.abs(mixer_attn), dim=-1)
            
            align_loss_raw = F.kl_div(mixer_log_prob, target_probs, reduction='none').sum(dim=-1)
            align_loss = (align_loss_raw * mask_wm).sum() / mask_wm.sum()
        else:
            align_loss = th.tensor(0.0).to(self.device)
        
        # Alignment Beta Annealing (线性衰减)
        # 目的: 初期强引导, 后期让 RL 自适应
        if t_env < getattr(self.args, "align_start_steps", 0):
            curr_beta = 0.0
        else:
            decay_steps = 2000000 
            progress = (t_env - self.args.align_start_steps) / decay_steps
            decay_factor = max(0.0, 1.0 - progress)
            curr_beta = self.causal_beta * decay_factor

        # =================================================================
        # 4. Optimization
        # =================================================================
        with th.no_grad():
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], z_tensor_mixer)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

        td_error = (mix_out - targets.detach())
        if td_error.shape[1] != mask.shape[1]:
             td_error = td_error[:, :min_len]
             mask = mask[:, :min_len]
             
        loss_td = (td_error ** 2 * mask.expand_as(td_error)).sum() / mask.sum()
        
        total_loss = loss_td + \
                     self.args.wm_loss_weight * wm_loss + \
                     curr_beta * align_loss

        self.optimiser.zero_grad()
        self.wm_optimiser.zero_grad()
        total_loss.backward()
        
        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        th.nn.utils.clip_grad_norm_(self.wm_params, self.args.grad_norm_clip)
        
        self.optimiser.step()
        self.wm_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_wm", wm_loss.item(), t_env)
            self.logger.log_stat("loss_causal", align_loss.item(), t_env)
            self.logger.log_stat("causal_beta", curr_beta, t_env)
            self.log_stats_t = t_env

    # 包含样板代码 (evaluate, cuda, save, load)
    def evaluate_world_model(self, batch, t_env):
        # 1. 数据准备
        if batch.device != self.args.device:
            batch.to(self.args.device)
            
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        # 构建输入: Global State + Joint Actions
        joint_actions = batch["actions_onehot"].reshape(bs, max_t, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1)
        
        # 真实标签 (从 t=1 到 t=T 的状态)
        target_states = batch["state"][:, 1:] 
        mask = batch["filled"][:, :-1].float().squeeze(-1) # [B, T-1] 用于过滤填充数据

        # 2. 切换评估模式
        self.world_model.eval()
        
        pred_states_list = []
        
        with th.no_grad():
            wm_hidden = self.world_model.init_hidden(bs, self.device)
            
            # 循环预测 t=0 到 t=T-1
            for t in range(max_t - 1):
                # 使用 predict 接口
                pred_next_state, wm_hidden = self.world_model.predict(
                    wm_inputs[:, t], wm_hidden, use_mean=True
                )
                pred_states_list.append(pred_next_state)
            
            # 堆叠预测结果 [B, T-1, State_Dim]
            pred_states = th.stack(pred_states_list, dim=1)

            # 3. 计算指标
            # (1) MSE (Mean Squared Error)
            errors = (pred_states - target_states) ** 2
            # 对 State 维度求和/平均 -> [B, T-1]
            mse_per_step = errors.mean(dim=-1) 
            # 应用 Mask (只计算有效时间步)
            masked_mse = (mse_per_step * mask).sum() / mask.sum()
            
            # (2) R^2 Score (Coefficient of Determination)
            # R2 = 1 - (SS_res / SS_tot)
            # SS_res = sum((y_true - y_pred)^2)
            # SS_tot = sum((y_true - y_mean)^2)
            
            # 将 Tensor 展平以便计算全局 R2
            mask_bool = mask.bool() # [B, T-1]
            # 只取有效数据
            y_true_flat = target_states[mask_bool.unsqueeze(-1).expand_as(target_states)].reshape(-1, self.state_dim)
            y_pred_flat = pred_states[mask_bool.unsqueeze(-1).expand_as(pred_states)].reshape(-1, self.state_dim)
            
            ss_res = ((y_true_flat - y_pred_flat) ** 2).sum()
            y_mean = y_true_flat.mean(dim=0) # 对每个状态特征维度求均值
            ss_tot = ((y_true_flat - y_mean) ** 2).sum()
            
            # 避免除以零
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

        # 恢复训练模式
        self.world_model.train() 

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
        try:
            self.wm_optimiser.load_state_dict(th.load("{}/wm_opt.th".format(path), map_location=lambda storage, loc: storage))
        except:
            pass