import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dvd_wm_causal_mixer import DVDWMCausalMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F

class DVDWMCausalLearner:
    def __init__(self, mac, scheme, logger, args, wm_learner=None):
        """
        wm_learner: 外部传入的 WMLearner 实例 (引用)
        """
        self.args = args
        self.mac = mac
        self.logger = logger
        self.wm_learner = wm_learner # 关键：持有 WM 引用
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # Mixer 初始化
        self.mixer = DVDWMCausalMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)

        # 优化器只负责 RL 参数 (MAC + Mixer)
        self.rl_params = list(mac.parameters()) + list(self.mixer.parameters())

        if args.optimizer == 'adam':
            self.optimiser = Adam(params=self.rl_params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.rl_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.causal_beta = getattr(args, "causal_beta", 0.01)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 1. RL Agent Forward
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Target MAC
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

        # 2. 调用 WM 获取因果权重 (Inference Only)
        # ----------------------------------------------------
        causal_weights_list, z_tensor = self.wm_learner.get_causal_weights(batch)
        
        # 处理 Z Warmup
        warmup_coef = 1.0 
        if getattr(self.args, "use_z_warmup", False) and t_env < self.args.z_warmup_steps:
             warmup_coef = float(t_env) / float(self.args.z_warmup_steps)
        z_eval = z_tensor[:, :, :-1] * warmup_coef

        # 3. Mixer Forward
        mix_out = self.mixer(chosen_action_qvals, batch["state"][:, :-1], z_eval)

        # 4. 计算 Alignment Loss
        if len(causal_weights_list) > 0:
            target_probs = th.stack(causal_weights_list, dim=1) # [B, T, N]
            target_probs = th.clamp(target_probs, min=1e-6, max=1.0-1e-6)
            
            # 这里的 Mixer Attention 获取方式取决于您的 Mixer 实现
            mixer_raw = self.mixer.hyper_attention.last_attn_weights
            d_samples = self.args.dvd_samples
            mixer_attn = mixer_raw.view(d_samples, bs, -1, self.args.n_agents).mean(dim=0)
            mixer_attn = mixer_attn[:, :target_probs.shape[1], :]
            
            mixer_log_prob = F.log_softmax(th.abs(mixer_attn), dim=-1)
            
            # Mask Alignment Loss (只计算 episode 存活期间)
            mask_wm = mask[:, :target_probs.shape[1]].squeeze(-1)
            align_loss_raw = F.kl_div(mixer_log_prob, target_probs, reduction='none').sum(dim=-1)
            align_loss = (align_loss_raw * mask_wm).sum() / mask_wm.sum()
        else:
            align_loss = th.tensor(0.0).to(self.device)

        # 5. 计算 TD Loss 与 Total Loss
        # Beta Annealing
        decay_steps = 2000000 
        if t_env < getattr(self.args, "align_start_steps", 0):
             curr_beta = 0.0
        else:
             progress = (t_env - self.args.align_start_steps) / decay_steps
             decay_factor = max(0.0, 1.0 - progress)
             curr_beta = self.causal_beta * decay_factor

        with th.no_grad():
            # Target Mixer 也需要 Z (可以使用相同的 z_eval)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], z_tensor[:, :, :] * warmup_coef)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)

        td_error = (mix_out - targets.detach())
        # Truncate
        min_len = min(td_error.shape[1], mask.shape[1])
        td_error = td_error[:, :min_len]
        mask = mask[:, :min_len]
        
        loss_td = (td_error ** 2 * mask.expand_as(td_error)).sum() / mask.sum()
        
        total_loss = loss_td + curr_beta * align_loss

        # 6. 优化
        self.optimiser.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.rl_params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update Targets
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss_td.item(), t_env)
            self.logger.log_stat("loss_align", align_loss.item(), t_env)
            self.logger.log_stat("beta", curr_beta, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        # WM 不需要在这里 cuda，因为它自己管理
        
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))