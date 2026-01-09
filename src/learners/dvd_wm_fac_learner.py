import copy
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_fac_mixer import DVDWMFacMixer
from modules.world_models.vae_rnn_fac import FactorizedVAERNN
from utils.rl_utils import build_td_lambda_targets

class DVDWMFacLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # 1. 初始化 Mixer
        self.mixer = None
        if args.mixer == "dvd_wm_fac":
            self.mixer = DVDWMFacMixer(args)
        else:
            raise ValueError(f"Mixer {args.mixer} not recognised for DVDWMFacLearner.")
        
        self.target_mixer = copy.deepcopy(self.mixer)

        # 2. 初始化 World Model
        self.obs_dim = scheme["obs"]["vshape"]
        self.act_dim = scheme["actions_onehot"]["vshape"][0]
        
        input_shape = self.obs_dim + self.act_dim
        output_shape = self.obs_dim
        
        self.world_model = FactorizedVAERNN(args, input_shape, output_shape)
        self.world_model.to(self.device)

        # 3. 定义优化器
        self.rl_params = list(mac.parameters()) + list(self.mixer.parameters())
        self.wm_params = list(self.world_model.parameters())

        # RL Optimizer
        if args.optimizer == 'adam':
            self.optimiser = Adam(params=self.rl_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.rl_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # WM Optimizer (通常 Adam 效果更好)
        self.wm_optimiser = Adam(params=self.wm_params, lr=args.wm_lr if hasattr(args, 'wm_lr') else args.lr)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # ----------------------------------------------------------------------
        # Part A: World Model Forward (Dynamics Learning)
        # ----------------------------------------------------------------------
        # 准备数据 [Batch, Seq, N, Dim]
        full_obs = batch["obs"].to(self.device)
        full_actions = batch["actions_onehot"].to(self.device)
        
        # WM 前向传播
        # 注意：z_all 现在是 [Batch, Seq, N, 64] (保留了 Agent 维度)
        z_all, recon_all, mu, logvar, _ = self.world_model(full_obs, full_actions)
        
        # 计算 WM Loss (预测未来: Input[:,-1] -> Target[:,1:])
        obs_input = full_obs[:, :-1]
        act_input = full_actions[:, :-1]
        target_obs = full_obs[:, 1:]
        
        # 重新跑一次切片后的前向，保证 Loss 计算严谨
        # 这里 recon_out 是 [Batch, Seq-1, N, Obs_Dim]
        _, recon_out, mu_train, logvar_train, _ = self.world_model(obs_input, act_input)
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_out, target_obs, reduction='none')
        recon_loss = recon_loss.sum(dim=-1).mean() # Sum over features, Mean over rest
        
        # 2. KL Divergence
        kld_loss = -0.5 * th.sum(1 + logvar_train - mu_train.pow(2) - logvar_train.exp(), dim=-1)
        kld_loss = kld_loss.mean()
        
        # [修复] 使用正确的参数名 wm_kl_beta
        beta = getattr(self.args, "wm_kl_beta", getattr(self.args, "beta", 0.1))
        wm_loss = recon_loss + beta * kld_loss

        # ----------------------------------------------------------------------
        # Part B: Prepare Z for Mixer (Deconfounding)
        # ----------------------------------------------------------------------
        # 切片 Z，分别用于 Q_eval 和 Target_Q
        # z_all: [B, T, N, Z]
        z_curr = z_all[:, :-1].detach() # 对应 t=0 ~ T-1
        z_next = z_all[:, 1:].detach()  # 对应 t=1 ~ T
        
        # Z-Warmup (可选)
        warmup_coef = 1.0
        if getattr(self.args, "use_z_warmup", False):
            if t_env < self.args.z_warmup_steps:
                warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        
        z_curr = z_curr * warmup_coef
        z_next = z_next * warmup_coef

        # ----------------------------------------------------------------------
        # Part C: RL Training (Q-Learning)
        # ----------------------------------------------------------------------
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 1. Calculate Estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1) # [B, T, N, A]
        
        # Pick Q-values for chosen actions
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # 2. Calculate Target Q-Values (Double Q-Learning)
        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = []
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)
            
            # Mask out unavailable actions
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            
            # Greedy action selection
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out[:, 1:], dim=3, index=cur_max_actions).squeeze(3)
            
            # Target Mixer: 传入 State[:, 1:] 和 z_next (Agent-wise Z)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], z_next)

        # 3. Mixer: 传入 State[:, :-1] 和 z_curr (Agent-wise Z)
        # Mixer 内部会处理 [B, T, N, Z] 的维度
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_curr)

        # 4. TD Error
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                        self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # [核心修复] 强制对齐维度，防止 83 vs 82 报错
        # 取两者最小的时间步数进行截断
        min_t = min(chosen_action_qvals.shape[1], targets.shape[1])
        chosen_action_qvals = chosen_action_qvals[:, :min_t]
        targets = targets[:, :min_t]
        mask = mask[:, :min_t]

        td_error = (chosen_action_qvals - targets.detach())
        
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask.expand_as(td_error)
        loss_td = (masked_td_error ** 2).sum() / mask.sum()

        # ----------------------------------------------------------------------
        # Part D: Optimization
        # ----------------------------------------------------------------------
        total_loss = loss_td + getattr(self.args, "wm_loss_weight", 1.0) * wm_loss

        self.optimiser.zero_grad()
        self.wm_optimiser.zero_grad()

        total_loss.backward()

        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        th.nn.utils.clip_grad_norm_(self.wm_params, self.args.grad_norm_clip)

        self.optimiser.step()
        self.wm_optimiser.step()

        # ----------------------------------------------------------------------
        # Part E: Logging & Updates
        # ----------------------------------------------------------------------
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_wm", wm_loss.item(), t_env)
            self.logger.log_stat("wm_recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat("wm_kld_loss", kld_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def evaluate_world_model(self, batch, t_env):
        """
        评估 WM 性能: 局部观测重构误差 MSE
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions_onehot"].to(self.device)
        mask = batch["filled"][:, :-1].float().squeeze(-1)
        
        obs_input = obs[:, :-1]
        act_input = actions[:, :-1]
        target_obs = obs[:, 1:]

        self.world_model.eval()
        
        with th.no_grad():
            _, recon_out, _, _, _ = self.world_model(obs_input, act_input)
            
            # 计算 MSE
            errors = (recon_out - target_obs) ** 2
            mse_per_step = errors.mean(dim=-1).mean(dim=-1) # Mean over Obs & Agents
            
            masked_mse = (mse_per_step * mask).sum() / mask.sum()

            # 简单的 R2 计算
            target_flat = target_obs.reshape(-1, self.obs_dim)
            pred_flat = recon_out.reshape(-1, self.obs_dim)
            ss_res = ((target_flat - pred_flat) ** 2).sum()
            ss_tot = ((target_flat - target_flat.mean(0)) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            log_prefix = "test_wm_"
            self.logger.log_stat(log_prefix + "mse", masked_mse.item(), t_env)
            self.logger.log_stat(log_prefix + "r2", r2.item(), t_env)
            
            print(f"\n[Test WM] MSE: {masked_mse.item():.6f} | R2: {r2.item():.6f}")

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
        th.save(self.world_model.state_dict(), "{}/world_model.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.wm_optimiser.load_state_dict(th.load("{}/wm_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.world_model.load_state_dict(th.load("{}/world_model.th".format(path), map_location=lambda storage, loc: storage))