import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from modules.world_models import REGISTRY as wm_REGISTRY

class WMLearner:
    def __init__(self, args):
        self.args = args
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        
        # 1. 初始化 World Model (兼容 VAE-RNN 或未来的 RSSM)
        # 根据 args.world_model_type 选择模型
        self.state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(args.state_shape[0])
        wm_in_shape = self.state_dim + (args.n_agents * args.n_actions)
        
        # 这里预留了扩展 RSSM 的接口，只要 RSSM 遵循相同的 forward/loss 接口即可
        self.world_model = wm_REGISTRY[args.world_model_type](wm_in_shape, args)
        
        self.params = list(self.world_model.parameters())
        
        # WM 通常推荐使用 Adam 优化器
        self.optimiser = Adam(params=self.params, lr=args.wm_lr, weight_decay=getattr(args, "weight_decay", 0))

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch, t_env, logger):
        """
        WM 的训练逻辑：包含 Full Path 和 Masked Path (Action Dropout)
        """
        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        
        # 确保数据在设备上
        batch.to(self.device)
        
        # 构造输入
        joint_actions = batch["actions_onehot"].reshape(bs, max_seq_length, -1)
        
        # [Path A]: Full Context Input
        wm_inputs_full = th.cat([batch["state"], joint_actions], dim=-1)
        
        # [Path B]: Masked Context Input (关键：模拟 ICES 的训练，让 WM 适应 Zero Mask)
        with th.no_grad():
            # 随机 Mask 掉一个 Agent 的动作 (置为0)
            rand_agent_idx = th.randint(0, self.args.n_agents, (bs, max_seq_length, 1), device=self.device)
            dropout_mask = 1.0 - F.one_hot(rand_agent_idx.squeeze(-1), num_classes=self.args.n_agents).float()
            dropout_mask_exp = dropout_mask.unsqueeze(-1).expand(bs, max_seq_length, self.args.n_agents, self.args.n_actions)
            
            joint_actions_masked = batch["actions_onehot"] * dropout_mask_exp
            joint_actions_masked_flat = joint_actions_masked.reshape(bs, max_seq_length, -1)
            
        wm_inputs_masked = th.cat([batch["state"], joint_actions_masked_flat], dim=-1)

        # 初始化 Hidden States (轨迹完整性保证：从 t=0 开始初始化)
        wm_hidden_full = self.world_model.init_hidden(bs, self.device)
        wm_hidden_masked = self.world_model.init_hidden(bs, self.device)
        
        recon_losses, kl_losses, reward_losses = [], [], []

        # ---------------------------------------------------
        # 时序前向传播 (RNN Unroll)
        # ---------------------------------------------------
        for t in range(max_seq_length):
            # 1. Full Path Forward
            input_full = wm_inputs_full[:, t]
            wm_hidden_full, dist_params_full = self.world_model.forward_step(input_full, wm_hidden_full)
            z_samples_full = self.world_model.sample_latents(dist_params_full, num_samples=1)

            # 2. Masked Path Forward
            input_masked = wm_inputs_masked[:, t]
            wm_hidden_masked, dist_params_masked = self.world_model.forward_step(input_masked, wm_hidden_masked)
            z_samples_masked = self.world_model.sample_latents(dist_params_masked, num_samples=1)

            # 3. Loss Calculation
            if t < max_seq_length - 1:
                target_state = batch["state"][:, t+1]
                target_reward = batch["reward"][:, t]
                mask = batch["filled"][:, t] # [B, 1]
                
                # [关键修正] 确保 mask 变成 [B] 以便与 Loss [B] 相乘，避免广播错误
                mask = mask.squeeze(-1) 
                
                # Loss A (Full)
                r_l_f, k_l_f, rew_l_f = self.world_model.compute_loss(
                    z_samples_full, wm_hidden_full, target_state, dist_params_full, target_reward
                )
                
                # Loss B (Masked)
                r_l_m, k_l_m, rew_l_m = self.world_model.compute_loss(
                    z_samples_masked, wm_hidden_masked, target_state, dist_params_masked, target_reward
                )
                
                # 直接相加并应用 Mask (此时大家都是 [B])
                recon_losses.append((r_l_f + r_l_m) * mask)
                kl_losses.append((k_l_f + k_l_m) * mask)
                reward_losses.append((rew_l_f + rew_l_m) * mask)

        # ---------------------------------------------------
        # 聚合 Loss 与 优化
        # ---------------------------------------------------
        # 将 list stack 起来 -> [T, B] -> sum -> scalar
        recon_loss = th.stack(recon_losses).sum() / batch["filled"].sum()
        kl_loss = th.stack(kl_losses).sum() / batch["filled"].sum()
        reward_loss = th.stack(reward_losses).sum() / batch["filled"].sum()

        total_loss = recon_loss + self.args.wm_kl_beta * kl_loss + self.args.wm_loss_weight * reward_loss

        self.optimiser.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            logger.log_stat("wm_loss", total_loss.item(), t_env)
            logger.log_stat("wm_recon_loss", recon_loss.item(), t_env)
            logger.log_stat("wm_reward_loss", reward_loss.item(), t_env)
            self.log_stats_t = t_env

    def get_causal_weights(self, batch):
        """
        供 RL Learner 调用的接口。
        执行因果推理，计算 (KL + Reward_Diff)。
        使用 Zero Mask，因为 train 中已经适应了。
        """
        self.world_model.eval() # 切换到 Eval 模式
        
        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        n_agents = self.args.n_agents
        
        # 准备 Full Path 的输入
        joint_actions = batch["actions_onehot"].reshape(bs, max_seq_length, -1)
        wm_inputs_full = th.cat([batch["state"], joint_actions], dim=-1)
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        causal_weights_history = []
        reward_sensitivity = getattr(self.args, "reward_sensitivity", 20.0)
        
        # 缓存 Z (用于 Mixer 输入)
        z_history = []

        # 轨迹遍历
        for t in range(max_seq_length):
            current_hidden = wm_hidden.detach().clone()
            
            # Factual Forward
            input_t = wm_inputs_full[:, t]
            wm_hidden, dist_params = self.world_model.forward_step(input_t, wm_hidden)
            
            # 采样 Z (用于 Mixer)
            z_samples = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_samples)

            # Causal Inference Loop (只计算有效时间步)
            if t < max_seq_length - 1:
                with th.no_grad():
                    # 1. 扩展数据
                    hidden_repeat = current_hidden.repeat_interleave(n_agents, dim=0) # [B*N, H]
                    state_t = batch["state"][:, t]
                    state_repeat = state_t.repeat_interleave(n_agents, dim=0)
                    
                    # 2. 构造 Zero Mask 动作
                    action_t = batch["actions_onehot"][:, t]
                    action_exp = action_t.unsqueeze(1).expand(-1, n_agents, -1, -1).clone() # [B, N, N, A]
                    
                    # 对角线置 0 (Masking)
                    mask_mat = 1.0 - th.eye(n_agents, device=self.device).view(1, n_agents, n_agents, 1)
                    # 结果: [B, N, N, A] -> [B*N, N*A] (Batch*Agent 被 Mask, Action Flat)
                    action_cf = (action_exp * mask_mat).view(bs * n_agents, -1)
                    
                    # 3. Counterfactual Forward (Zero Mask)
                    pert_input = th.cat([state_repeat, action_cf], dim=-1)
                    hidden_next_cf, dist_cf = self.world_model.forward_step(pert_input, hidden_repeat)
                    
                    # 4. 计算 Hybrid Effect (KL + Reward Diff)
                    # 传入的 dist_params 是 facts (h_next), dist_cf 是 counterfactual (h_next_cf)
                    # 注意: 为了计算 Reward Diff，我们还需要 factual 的 next_hidden (wm_hidden)
                    total_effect = self._compute_hybrid_effect(
                        dist_params, dist_cf, wm_hidden, hidden_next_cf, n_agents, bs, reward_sensitivity
                    )
                    
                    causal_weights_history.append(F.softmax(total_effect, dim=-1))
        
        self.world_model.train() # 切回 Train 模式
        
        # 堆叠 Z
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach() # [D, B, T, L]
        
        return causal_weights_history, z_tensor

    def _compute_hybrid_effect(self, dist_fac, dist_cf, h_fac, h_cf, n_agents, bs, sensitivity):
        """
        计算因果效应：物理效应 (KL) + 价值效应 (Reward Diff)
        """
        # 1. 准备参数
        # Factual: [B, L] -> 扩展为 [B, N, L] 以便和 N 个反事实进行对比
        mu_fac = dist_fac[0].detach().unsqueeze(1).expand(-1, n_agents, -1)
        logvar_fac = dist_fac[1].detach().unsqueeze(1).expand(-1, n_agents, -1)
        var_fac = th.exp(logvar_fac)
        
        # Counterfactual: [B*N, L] -> Reshape 为 [B, N, L]
        mu_cf = dist_cf[0].view(bs, n_agents, -1)
        logvar_cf = dist_cf[1].view(bs, n_agents, -1)
        var_cf = th.exp(logvar_cf)
        
        # 2. 计算 KL 散度: KL(N_fac || N_cf)
        # 公式: 0.5 * ( (var1 / var2) + (mu1 - mu2)^2 / var2 + ln(var2/var1) - k )
        term1 = var_fac / (var_cf + 1e-8)
        term2 = (mu_fac - mu_cf).pow(2) / (var_cf + 1e-8)
        term3 = logvar_cf - logvar_fac
        # Sum over latent dimensions
        kl_div = 0.5 * th.sum(term1 + term2 + term3 - 1, dim=-1) # [B, N]
        
        # 3. 计算 Reward Difference
        # Factual Reward: Head(z_fac, h_fac)
        # z_fac: [B, L] -> Use Mean
        z_fac = dist_fac[0].detach() 
        # h_fac: [B, H]
        r_in_fac = th.cat([z_fac, h_fac.detach()], dim=-1)
        r_fac = self.world_model.reward_head(r_in_fac) # [B, 1]
        r_fac_exp = r_fac.unsqueeze(1).expand(-1, n_agents, -1) # [B, N, 1]
        
        # Counterfactual Reward: Head(z_cf, h_cf)
        # z_cf: [B*N, L]
        z_cf = dist_cf[0] 
        # h_cf: [B*N, H]
        r_in_cf = th.cat([z_cf, h_cf], dim=-1)
        r_cf = self.world_model.reward_head(r_in_cf) # [B*N, 1]
        r_cf_view = r_cf.view(bs, n_agents, -1) # [B, N, 1]
        
        # Squared Difference (L2)
        r_diff = (r_fac_exp - r_cf_view).pow(2).sum(dim=-1) # [B, N]
        
        # 4. 合并效应
        return kl_div + sensitivity * r_diff

    def cuda(self):
        self.world_model.cuda()
    
    def save_models(self, path):
        th.save(self.world_model.state_dict(), "{}/world_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/wm_opt.th".format(path))

    def load_models(self, path):
        self.world_model.load_state_dict(th.load("{}/world_model.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/wm_opt.th".format(path), map_location=lambda storage, loc: storage))