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
        self.state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(args.state_shape[0]) # 处理可能的shape格式
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
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        # ==================== World Rollout (Modified for RSSM) ====================
        bs = batch.batch_size
        
        # 1. 构建输入: Global State + Joint Actions
        joint_actions = batch["actions_onehot"].reshape(bs, batch.max_seq_length, -1)
        wm_inputs = th.cat([batch["state"], joint_actions], dim=-1) # [B, T, State + N*A]
        
        # 初始化 Hidden State
        wm_hidden = self.world_model.init_hidden(bs, self.device)
        
        z_history = []          # 用于存储传给 Mixer 的 z
        forward_outputs = []    # [新增] 用于存储 RSSM 的中间输出 (h, post, prior 等) 以便计算 Loss

        # 循环 t
        for t in range(batch.max_seq_length):
            # 1. WM Step (Forward)
            # RSSM 的 forward_step 返回: next_hidden, dist_params
            # 注意: 这里的 dist_params 包含了 (prior, post, z) 等复杂结构
            wm_hidden, dist_params = self.world_model.forward_step(wm_inputs[:, t], wm_hidden)
            
            # 2. 收集输出供后续计算 Loss
            forward_outputs.append((wm_hidden, dist_params))

            # 3. Sample Z for Mixer (使用 sample_latents 接口适配)
            # 这里的 z_samples 已经是拼接好 [h, z] 的特征了 (如果使用我之前给的 RSSM 代码)
            z_samples = self.world_model.sample_latents(dist_params, num_samples=self.args.dvd_samples)
            z_history.append(z_samples)

        # ==================== World Model Loss Calculation (Batch) ====================
        # 这里的逻辑发生了变化：我们在循环外统一计算 Loss
        
        # 准备真实标签 (Targets)
        # 我们用 t=0 的输入预测 t=1 的状态，所以 Target 是 state[:, 1:]
        # forward_outputs 包含 t=0 到 t=T-1 的输出 (共 T 个)
        # 但有效的预测通常直到 T-1 (因为 batch["state"] 长度为 T)
        # 确保对齐：取前 T-1 个输出，对应 batch["state"][:, 1:]
        
        seq_len = batch.max_seq_length
        if seq_len > 1:
            # 取出前 T-1 个时间步的预测输出
            valid_outputs = forward_outputs[:-1] 
            
            # 对应的真实标签: State_{1} 到 State_{T-1}
            target_states = batch["state"][:, 1:]
            
            # 对应的真实奖励: Reward_{0} 到 Reward_{T-2} (PyMARL 的 reward 对齐比较特殊，通常 r_t 是 s_t 采取 a_t 后的奖励)
            # batch["reward"][:, :-1] 对应 0 到 T-2
            target_rewards = batch["reward"][:, :-1]

            # 统一计算 Loss
            # RSSM 的 compute_loss 内部处理时间维度
            recon_loss, kl_loss, reward_loss = self.world_model.compute_loss(
                valid_outputs, target_states, target_rewards
            )
            
            # 应用 Mask (mask 的维度通常是 [B, T-1])
            mask_squeeze = mask.squeeze(-1) 
            
            # 确保 loss 维度与 mask 一致 [B, T-1]
            # 如果 compute_loss 返回的是 [B, T-1]，则直接相乘
            
            wm_loss = (recon_loss * mask_squeeze).sum() / mask_squeeze.sum() + \
                      self.args.wm_kl_beta * (kl_loss * mask_squeeze).sum() / mask_squeeze.sum() + \
                      (reward_loss * mask_squeeze).sum() / mask_squeeze.sum()
        else:
            wm_loss = th.tensor(0.0).to(self.device)
        # ==================== Mixing & RL Loss ====================
        # Prepare Z
        # z_history list: T * [D, B, L]
        # Stack -> [T, D, B, L] -> Permute -> [D, B, T, L]
        # 注意: 这里不再有 Agent 维度，因为 Z 是 Global 的
        # z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3)
        # [修改后] 添加 .detach()
        # 解释: .detach() 会创建一个新的 Tensor，其数值相同但 requires_grad=False。
        # 这样 loss_td.backward() 更新 Mixer 参数时，梯度传到这里就会停止，不会流向 World Model。
        z_tensor = th.stack(z_history, dim=0).permute(1, 2, 0, 3).detach()
        
        
        # === 修改 3: Z-Warmup 实现 (Scheme 3) ===
        # 默认不缩放
        warmup_coef = 1.0 
        
        if getattr(self.args, "use_z_warmup", False):
            if t_env < self.args.z_warmup_steps:
                warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
            else:
                warmup_coef = 1.0
        
        # 应用 Warmup 系数到传入 Mixer 的 Z 上
        # 注意：这不会影响 World Model 的 loss 计算 (KL/Recon)，只影响 Mixer 的输入
        z_tensor_mixer = z_tensor * warmup_coef

        z_eval = z_tensor_mixer[:, :, :-1]
        z_target = z_tensor_mixer[:, :, 1:]

        # Mixer Forward (State + Z)
        # 这里的 z_eval 已经是 detached 的了，所以 Mixer 的梯度不会传给 WM
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)
        
        
        with th.no_grad():
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], z_target)

            # [修复后代码] 
            # 对 target_max_qvals 进行切片 [:, 1:]，对应 t=1 到 t=T，与 state[:, 1:] 对齐
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], z_tensor_mixer)
            
            if getattr(self.args, 'q_lambda', False):
                 pass 
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

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