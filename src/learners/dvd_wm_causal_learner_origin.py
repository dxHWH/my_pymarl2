import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_causal_mixer import DVDWMCausalMixer
from modules.world_models import REGISTRY as wm_REGISTRY
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class DVDWMCausalLearner_origin:
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
        # 1. 基础配置与数据准备
        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # 优化器配置 (建议使用 Adam)
        # reward_sensitivity: 建议 20.0 ~ 50.0，平衡 KL 和 Reward 量级
        reward_sensitivity = getattr(self.args, "reward_sensitivity", 20.0)

        # =================================================================
        # 2. RL Agent (MAC) Forward
        # =================================================================
        # 标准 QMIX 流程，无需改动
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

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
        # 3. World Model Training (Dual-Path: Full & Masked) [核心重构]
        # =================================================================
        # 目的：让 WM 既能理解完整世界，也能理解缺失 Agent 的世界(Zero Mask)
        
        joint_actions = batch["actions_onehot"].reshape(bs, max_seq_length, -1)
        
        # [Path A]: Full Context Input
        wm_inputs_full = th.cat([batch["state"], joint_actions], dim=-1)
        
        # [Path B]: Masked Context Input (模拟 ICES 的 Local Model 训练)
        # 策略：每个样本在每一时刻，随机 Mask 掉一个 Agent 的动作 (置为0)
        with th.no_grad():
            # 生成随机索引 [B, T, 1]
            rand_agent_idx = th.randint(0, self.args.n_agents, (bs, max_seq_length, 1), device=self.device)
            # 生成 Mask 矩阵 [B, T, N] (1=Keep, 0=Drop)
            dropout_mask = 1.0 - F.one_hot(rand_agent_idx.squeeze(-1), num_classes=self.args.n_agents).float()
            # 扩展到动作维度 [B, T, N, A]
            dropout_mask_exp = dropout_mask.unsqueeze(-1).expand(bs, max_seq_length, self.args.n_agents, self.args.n_actions)
            # 应用 Mask (Zero Masking)
            joint_actions_masked = batch["actions_onehot"] * dropout_mask_exp
            joint_actions_masked_flat = joint_actions_masked.reshape(bs, max_seq_length, -1)
            
        wm_inputs_masked = th.cat([batch["state"], joint_actions_masked_flat], dim=-1)

        # 初始化 Hidden States
        wm_hidden_full = self.world_model.init_hidden(bs, self.device)
        wm_hidden_masked = self.world_model.init_hidden(bs, self.device) # Masked 路径需要独立的 hidden 演化
        
        recon_losses, kl_losses, reward_losses = [], [], []
        
        # 存储用于 RL 的 Latent (只使用 Full Path 的 z，因为它最准确)
        z_history_rl = []
        # 存储用于因果推理的 Hidden (Full Path)
        hidden_history_causal = []

        for t in range(max_seq_length):
            # 缓存当前 Full Hidden 用于后面的 Causal Inference
            current_hidden_full = wm_hidden_full.detach().clone()
            hidden_history_causal.append(current_hidden_full)

            # --- Forward Path A (Full) ---
            input_full = wm_inputs_full[:, t]
            wm_hidden_full, dist_params_full = self.world_model.forward_step(input_full, wm_hidden_full)
            z_samples_full = self.world_model.sample_latents(dist_params_full, num_samples=self.args.dvd_samples)
            z_history_rl.append(z_samples_full)

            # --- Forward Path B (Masked) ---
            # 这一步是为了训练 WM 适应 Zero Mask
            input_masked = wm_inputs_masked[:, t]
            wm_hidden_masked, dist_params_masked = self.world_model.forward_step(input_masked, wm_hidden_masked)
            z_samples_masked = self.world_model.sample_latents(dist_params_masked, num_samples=1) # 训练不需要多采样

            # --- Compute Losses ---
            if t < max_seq_length - 1:
                target_state = batch["state"][:, t+1]
                target_reward = batch["reward"][:, t]
                
                # Loss A: Full Context 重建能力
                r_l_f, k_l_f, rew_l_f = self.world_model.compute_loss(
                    z_samples_full, wm_hidden_full, target_state, dist_params_full, target_reward
                )
                
                # Loss B: Masked Context 重建能力 (关键！)
                # 强迫模型利用剩下的 N-1 个 Agent 还原真实结果
                r_l_m, k_l_m, rew_l_m = self.world_model.compute_loss(
                    z_samples_masked, wm_hidden_masked, target_state, dist_params_masked, target_reward
                )
                
                # 合并 Loss (取平均或加和)
                recon_losses.append(r_l_f + r_l_m)
                kl_losses.append(k_l_f + k_l_m)
                reward_losses.append(rew_l_f + rew_l_m)

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
        # 4. Causal Inference (基于 Zero Mask)
        # =================================================================
        # 现在 WM 已经见过 Zero Mask 了，我们可以放心地使用全零向量进行反事实推理
        
        causal_weights_history = [] 
        
        # 只需要计算到 min_len
        for t in range(min_len):
            with th.no_grad():
                n_agents = self.args.n_agents
                
                # 1. 准备数据
                # Factual Hidden (h_t)
                hidden_t = hidden_history_causal[t] # [B, H]
                
                # State & Action (t)
                state_t = batch["state"][:, t]
                action_t = batch["actions_onehot"][:, t] # [B, N, A]
                
                # 扩展数据 [B*N, ...]
                state_repeat = state_t.repeat_interleave(n_agents, dim=0)
                hidden_repeat = hidden_t.repeat_interleave(n_agents, dim=0)
                
                # 2. 构造反事实动作: Zero Masking (Theory Correct & Now In-Distribution)
                # ------------------------------------------------------
                action_exp = action_t.unsqueeze(1).expand(-1, n_agents, -1, -1).clone() # [B, N, N, A]
                
                # 构造对角线 Mask (保留非对角线，对角线置 0)
                mask_mat = 1.0 - th.eye(n_agents, device=self.device).view(1, n_agents, n_agents, 1)
                
                # 直接乘！对角线变为 [0,0,0...]
                action_cf_4d = action_exp * mask_mat 
                action_cf_flat = action_cf_4d.view(bs * n_agents, -1)
                
                # 3. WM Forward
                # Factual Pass (为了获取基准 Reward 和 KL 参数)
                wm_in_fac = wm_inputs_full[:, t]
                hidden_next_fac, dist_fac = self.world_model.forward_step(wm_in_fac, hidden_t)
                
                # Counterfactual Pass (Zero Mask)
                pert_input = th.cat([state_repeat, action_cf_flat], dim=-1)
                hidden_next_cf, dist_cf = self.world_model.forward_step(pert_input, hidden_repeat)
                
                # 4. 计算混合效应 (Hybrid Effect)
                # ------------------------------------------------------
                
                # (A) 物理效应 (KL)
                mu_fac = dist_fac[0].detach().unsqueeze(1).expand(-1, n_agents, -1)
                logvar_fac = dist_fac[1].detach().unsqueeze(1).expand(-1, n_agents, -1)
                var_fac = th.exp(logvar_fac)
                
                mu_cf = dist_cf[0].view(bs, n_agents, -1)
                logvar_cf = dist_cf[1].view(bs, n_agents, -1)
                var_cf = th.exp(logvar_cf)
                
                term1 = var_fac / (var_cf + 1e-8)
                term2 = (mu_fac - mu_cf).pow(2) / (var_cf + 1e-8)
                term3 = logvar_cf - logvar_fac
                kl_div = 0.5 * th.sum(term1 + term2 + term3 - 1, dim=-1)
                
                # (B) 价值效应 (Reward Diff)
                # Factual Reward: Head(z_fac, h_{t+1})
                z_fac = dist_fac[0].detach() 
                r_in_fac = th.cat([z_fac, hidden_next_fac.detach()], dim=-1)
                r_fac = self.world_model.reward_head(r_in_fac) # [B, 1]
                r_fac_exp = r_fac.unsqueeze(1).expand(-1, n_agents, -1)
                
                # Counterfactual Reward: Head(z_cf, h'_{t+1})
                # 由于 WM 见过 Zero Mask，它会预测出:"如果这个Agent不在，可能没伤害，没Reward"
                z_cf = dist_cf[0]
                r_in_cf = th.cat([z_cf, hidden_next_cf], dim=-1)
                r_cf = self.world_model.reward_head(r_in_cf) # [B*N, 1]
                r_cf_view = r_cf.view(bs, n_agents, -1)
                
                # Diff (L2 or L1)
                r_diff = (r_fac_exp - r_cf_view).pow(2).sum(dim=-1)
                
                # Combine
                total_effect = kl_div + reward_sensitivity * r_diff
                causal_weights_history.append(F.softmax(total_effect, dim=-1))

        # =================================================================
        # 5. Mixer Forward & Alignment
        # =================================================================
        z_tensor = th.stack(z_history_rl, dim=0).permute(1, 2, 0, 3).detach() # [D, B, T, L]
        
        # Warmup
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False) and t_env < self.args.z_warmup_steps:
             warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        z_tensor_mixer = z_tensor * warmup_coef
        z_eval = z_tensor_mixer[:, :, :-1]
        
        # Mixer Forward
        mix_out = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)

        # Alignment Loss
        if len(causal_weights_history) > 0:
            target_probs = th.stack(causal_weights_history, dim=1) # [B, T, N]
            target_probs = th.clamp(target_probs, min=1e-6, max=1.0-1e-6)
            
            mixer_raw = self.mixer.hyper_attention.last_attn_weights
            d_samples = self.args.dvd_samples
            # [D, B, T, N] -> [B, T, N]
            mixer_attn = mixer_raw.view(d_samples, bs, -1, self.args.n_agents).mean(dim=0)
            
            # Truncate to min_len
            valid_T = target_probs.shape[1]
            mixer_attn = mixer_attn[:, :valid_T, :]
            
            mixer_log_prob = F.log_softmax(th.abs(mixer_attn), dim=-1)
            align_loss_raw = F.kl_div(mixer_log_prob, target_probs, reduction='none').sum(dim=-1)
            align_loss = (align_loss_raw * mask_wm).sum() / mask_wm.sum()
        else:
            align_loss = th.tensor(0.0).to(self.device)
            
        # Annealing Beta
        if t_env < getattr(self.args, "align_start_steps", 0):
            curr_beta = 0.0
        else:
            decay_steps = 2000000 
            progress = (t_env - self.args.align_start_steps) / decay_steps
            decay_factor = max(0.0, 1.0 - progress)
            curr_beta = self.causal_beta * decay_factor

        # =================================================================
        # 6. Optimization
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