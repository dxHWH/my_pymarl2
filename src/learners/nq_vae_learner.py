import torch.optim as optim
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
import torch as th

from .nq_learner import NQLearner 
from world_model_for_key_agent.world.vae import WorldModelVAE, vae_loss_function 
from world_model_for_key_agent.world.res_vae import WorldModelResidualVAE

class NQLearnerVAE(NQLearner):
    """
    这个 Learner 继承自 NQLearner (或 QLearner),
    并添加了 "在线 VAE 训练" 和 "KL 退火" 的逻辑, 
    用于从零开始的端到端训练。
    """
    def __init__(self, mac, scheme, logger, args):
        # 1. 首先，初始化父类 (NQLearner)
        super(NQLearnerVAE, self).__init__(mac, scheme, logger, args)

        self.logger.console_logger.info("Initializing NQLearnerVAE (Online VAE Training Enabled)...")
        # (确保 training_steps 被父类初始化, 否则 self.training_steps = 0)

        # === [修复] 手动初始化计数器 ===
        self.training_steps = 0 
        # ===========================
        self.last_log_t = 0  # 用于 VAE 日志记录的时间戳
        # 2. VAE 在线训练初始化
        self.use_online_vae_training = getattr(self.args, "use_online_vae_training", False)
        if self.use_online_vae_training:
            self.logger.console_logger.info("Initializing VAE for ONLINE training FROM SCRATCH.")
            try:
                # 3. 从 scheme 获取维度
                obs_shape = scheme["obs"]["vshape"]
                n_agents = self.args.n_agents
                n_actions = self.args.n_actions
                state_shape = scheme["state"]["vshape"]
                
                vae_input_dim = (n_agents * obs_shape) + (n_agents * n_actions)
                
                # 4. (关键) 创建 *唯一* 的 VAE 模型实例
                self.vae_model = WorldModelVAE(
                    input_dim=vae_input_dim,
                    state_dim=state_shape,
                    latent_dim=self.args.vae_latent_dim,
                    hidden_dim=self.args.vae_hidden_dim
                ).to(self.device)
                self.vae_model.train() # 保持训练模式

            #     # [修改] 使用增强版 VAE，添加残差块与更深层
            #     self.vae_model = WorldModelResidualVAE(
            #         input_dim=vae_input_dim,
            #         state_dim=state_shape, 
            #         latent_dim=self.args.vae_latent_dim,
            #         hidden_dim=self.args.vae_hidden_dim,    # 这里会自动读取到 1024
            #         num_res_blocks=3                       
            # ).to(self.args.device)
            #     self.vae_model.train() # 保持训练模式
                
                # 5. 为 VAE 创建优化器
                self.vae_optimizer = optim.Adam(
                    self.vae_model.parameters(), 
                    lr=self.args.vae_lr, 
                    eps=self.args.optim_eps
                )
                
                # 6. 存储 KL 退火参数
                self.vae_kl_anneal_start_t = getattr(self.args, "vae_kl_anneal_start_t", 50000)
                self.vae_kl_anneal_duration_t = getattr(self.args, "vae_kl_anneal_duration_t", 100000)
                self.vae_beta_max = getattr(self.args, "vae_beta_max", 1.0)
                
            except Exception as e:
                self.logger.console_logger.error(f"!!!!!!!! FAILED TO INITIALIZE VAE !!!!!!!!")
                self.logger.console_logger.error(f"Error: {e}")
                self.use_online_vae_training = False
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 1. 运行父类的 train 方法 (训练 QMIX)
        # (这会调用 self.optimiser.step() 并更新 QMIX)
        super(NQLearnerVAE, self).train(batch, t_env, episode_num)

        # === [修复] 计数器自增 ===
        self.training_steps += 1
        # =======================
        
        # 2. VAE 在线训练逻辑
        if self.use_online_vae_training and self.vae_model is not None:
            
            # 2a. 准备 VAE 数据 (o_t, a_t) -> s_t+1
            obs_t = batch["obs"][:, :-1]
            actions_t = batch["actions_onehot"][:, :-1] 
            state_t_plus_1 = batch["state"][:, 1:]
            
            bs, max_t, n_agents, _ = obs_t.shape
            
            # 2b. 准备 VAE 输入 (bs*T, input_dim)
            obs_t_flat_vae = obs_t.reshape(bs * max_t, -1)
            actions_t_flat_vae = actions_t.reshape(bs * max_t, -1)
            vae_input = th.cat([obs_t_flat_vae, actions_t_flat_vae], dim=1)
            
            # 2c. 准备 VAE 目标 (bs*T, state_dim)
            vae_target = state_t_plus_1.reshape(bs * max_t, -1)
            
            # 2d. VAE 前向传播
            recon_state, mu, log_var = self.vae_model(vae_input)
            
            # 2e. 计算 *逐元素* (per-element) 损失
            rce_per_element = F.mse_loss(recon_state, vae_target, reduction='none').sum(dim=1) 
            kld_per_element = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            
            # 2f. 计算动态 Beta (KL 退火)
            if t_env < self.vae_kl_anneal_start_t:
                current_beta = 0.0
            else:
                progress = (t_env - self.vae_kl_anneal_start_t) / self.vae_kl_anneal_duration_t
                current_beta = min(self.vae_beta_max, progress * self.vae_beta_max)
            
            # 2g. 计算总损失 (per-element)
            total_loss_per_element = rce_per_element + current_beta * kld_per_element
            
            # 2h. 准备掩码 (Masking)
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask = (mask - terminated).squeeze(-1) # (bs, T-1)
            mask = mask.reshape(-1) # (bs*T,)
            
            # 2i. 计算加权平均损失
            vae_loss_mean = (total_loss_per_element * mask).sum() / (mask.sum() + 1e-8)
            
            # 2j. VAE 反向传播
            self.vae_optimizer.zero_grad()
            vae_loss_mean.backward()
            th.nn.utils.clip_grad_norm_(self.vae_model.parameters(), self.args.grad_norm_clip)
            self.vae_optimizer.step()
            
            # 2k. 记录日志
            # if self.training_steps % self.args.learner_log_interval == 0:
            #     self.logger.log_stat("vae_current_beta", current_beta, t_env)
            #     self.logger.log_stat("vae_online_loss_mean", vae_loss_mean.item(), t_env)
            #     rce_mean = (rce_per_element * mask).sum().item() / (mask.sum() + 1e-8)
            #     kld_mean = (kld_per_element * mask).sum().item() / (mask.sum() + 1e-8)
            #     self.logger.log_stat("vae_online_rce_mean", rce_mean, t_env)
            #     self.logger.log_stat("vae_online_kld_mean", kld_mean, t_env)
            if t_env - self.last_log_t >= self.args.learner_log_interval:
                self.logger.log_stat("vae_current_beta", current_beta, t_env)
                self.logger.log_stat("vae_online_loss_mean", vae_loss_mean.item(), t_env)
                
                rce_mean = (rce_per_element * mask).sum().item() / (mask.sum() + 1e-8)
                kld_mean = (kld_per_element * mask).sum().item() / (mask.sum() + 1e-8)
                
                self.logger.log_stat("vae_online_rce_mean", rce_mean, t_env)
                self.logger.log_stat("vae_online_kld_mean", kld_mean, t_env)
                self.last_log_t = t_env # 更新上次记录时间