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
        max_seq_length = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        bs = batch.batch_size

        # --- 1. RL Agent Forward ---
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(bs)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # --- 2. World Model & Causal Perturbation ---
        joint_actions = batch["actions_onehot"].reshape(bs, max_seq_length, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1)
        
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        recon_losses, kl_losses, reward_losses = [], [], []
        z_history = []
        causal_weights_history = [] 

        for t in range(max_seq_length):
            # A. Factual Pass
            input_t = wm_inputs[:, t]
            wm_hidden, dist_params = self.world_model.forward_step(input_t, wm_hidden)
            
            # B. Causal Perturbation Pass (Run only for valid training steps)
            if t < max_seq_length - 1:
                with th.no_grad():
                    n_agents = self.args.n_agents
                    
                    # 1. 扩展输入以进行并行推断 [B*N, Input_Dim]
                    state_t = batch["state"][:, t] # [B, S]
                    state_repeat = state_t.repeat_interleave(n_agents, dim=0) # [B*N, S]
                    
                    # 2. 构造 Masked Actions
                    action_t = batch["actions_onehot"][:, t] # [B, N, A]
                    action_expanded = action_t.unsqueeze(1).expand(-1, n_agents, -1, -1).clone() # [B, N, N, A]
                    
                    # 对角线Mask: 在第 i 组中 Mask 掉 Agent i
                    mask_mat = 1.0 - th.eye(n_agents, device=self.device).view(1, n_agents, n_agents, 1)
                    action_masked = action_expanded * mask_mat
                    action_masked_flat = action_masked.view(bs * n_agents, -1) # [B*N, N*A]
                    
                    perturbed_input = th.cat([state_repeat, action_masked_flat], dim=-1)
                    
                    # 3. 扩展 Hidden
                    hidden_repeat = wm_hidden.detach().repeat_interleave(n_agents, dim=0)
                    
                    # 4. Counterfactual Forward
                    _, cf_dist_params = self.world_model.forward_step(perturbed_input, hidden_repeat)
                    
                    # C. 计算 KL 散度因果效应
                    # P (Factual): mu_fac, var_fac
                    mu_fac, logvar_fac = dist_params
                    mu_fac = mu_fac.detach().unsqueeze(1).expand(-1, n_agents, -1)
                    logvar_fac = logvar_fac.detach().unsqueeze(1).expand(-1, n_agents, -1)
                    var_fac = th.exp(logvar_fac)
                    
                    # Q (Counterfactual)
                    mu_cf, logvar_cf = cf_dist_params
                    mu_cf = mu_cf.view(bs, n_agents, -1)
                    logvar_cf = logvar_cf.view(bs, n_agents, -1)
                    var_cf = th.exp(logvar_cf)
                    
                    # KL(P||Q)
                    term1 = var_fac / (var_cf + 1e-8)
                    term2 = (mu_fac - mu_cf).pow(2) / (var_cf + 1e-8)
                    term3 = logvar_cf - logvar_fac
                    kl_div = 0.5 * th.sum(term1 + term2 + term3 - 1, dim=-1) # [B, N]
                    
                    # 归一化作为权重 Teacher
                    causal_weights = F.softmax(kl_div, dim=-1) 
                    causal_weights_history.append(causal_weights)

            # C. 常规采样与 Loss
            z_samples = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_samples)
            
            if t < max_seq_length - 1:
                target_state = batch["state"][:, t+1]
                target_reward = batch["reward"][:, t]
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

        # --- 3. Mixer Forward & Alignment ---
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach()
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False) and t_env < self.args.z_warmup_steps:
             warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        z_tensor_mixer = z_tensor * warmup_coef
        z_eval = z_tensor_mixer[:, :, :-1]
        
        mix_out = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)

        # === Alignment Logic ===
        # Teacher: Causal Weights [B, T-1, N]
        target_probs = th.stack(causal_weights_history, dim=1)
        target_probs = th.clamp(target_probs, min=1e-6, max=1.0-1e-6)
        
        # Student: Mixer Attention
        mixer_raw = self.mixer.hyper_attention.last_attn_weights
        d_samples = self.args.dvd_samples
        mixer_attn = mixer_raw.view(d_samples, bs, -1, self.args.n_agents).mean(dim=0)
        
        # 使用 log_softmax 保证数值稳定 (Input: abs(tanh))
        mixer_log_prob = F.log_softmax(th.abs(mixer_attn), dim=-1)
        
        # KL Loss
        align_loss_raw = F.kl_div(mixer_log_prob, target_probs, reduction='none').sum(dim=-1)
        align_loss = (align_loss_raw * mask_wm).sum() / mask_wm.sum()
        
        # Warmup
        current_beta = 0.0 if t_env < getattr(self.args, "align_start_steps", 0) else self.causal_beta

        # --- 4. Optimization ---
        with th.no_grad():
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], z_tensor_mixer)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

        td_error = (mix_out - targets.detach())
        if td_error.shape[1] != mask.shape[1]:
             min_len = min(td_error.shape[1], mask.shape[1])
             td_error = td_error[:, :min_len]
             mask = mask[:, :min_len]
             
        loss_td = (td_error ** 2 * mask.expand_as(td_error)).sum() / mask.sum()
        
        total_loss = loss_td + self.args.wm_loss_weight * wm_loss + current_beta * align_loss

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