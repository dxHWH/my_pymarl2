import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RSSMWorldModel(nn.Module):
    def __init__(self, input_dim, args):
        super(RSSMWorldModel, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')

        # === 1. 维度解析 ===
        # input_dim = State_Dim + (N_Agents * N_Actions)
        # 我们需要手动拆分，因为 RSSM 对 State 和 Action 的处理方式不同
        if isinstance(args.state_shape, int):
            self.state_dim = args.state_shape
        else:
            self.state_dim = int(args.state_shape[0])
            
        self.action_dim = args.n_agents * args.n_actions
        
        # 验证维度 (Learner 传入的 input_dim 应该是两者之和)
        assert input_dim == self.state_dim + self.action_dim, "RSSM Input dim mismatch!"

        # RSSM 专属维度配置 (需要在 yaml 中定义)
        self.hidden_dim = args.rssm_hidden_dim  # Deterministic h_t
        self.latent_dim = args.rssm_latent_dim  # Stochastic z_t
        self.embed_dim = getattr(args, "rssm_embed_dim", 256)
        self.min_std = 0.1

        # === 2. 网络组件 ===
        
        # (a) Encoder: 压缩 State 和 Action
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ELU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.ELU()
        )

        # (b) Recurrent Model (Deterministic Path): h_t -> h_{t+1}
        # Input: h_t, z_t, a_t
        # Update: h_{t+1} = GRU(h_t, concat(z_t, a_t))
        self.rnn_cell = nn.GRUCell(self.latent_dim + self.embed_dim, self.hidden_dim)

        # (c) Transition Model (Prior): p(z_t | h_t)
        # 仅基于历史记忆 h_t 预测当前的 z_t 分布
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 2 * self.latent_dim) # Mean + LogVar
        )

        # (d) Representation Model (Posterior): q(z_t | h_t, s_t)
        # 结合 历史记忆 h_t 和 当前观测 s_t 推断 z_t
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 2 * self.latent_dim) # Mean + LogVar
        )

        # (e) State Predictor (Decoder): Predict S_{t+1} from h_{t+1} (and z_{t+1} sampled from prior)
        # 简单起见，且为了高效，直接使用 h_{t+1} 预测 S_{t+1}
        # h_{t+1} 已经聚合了 (h_t, z_t, a_t)，包含了预测下一帧所需的所有信息
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.state_dim)
        )

        # (f) Reward Predictor
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, 1)
        )

    def init_hidden(self, batch_size, device):
        # 返回 RNN 的初始状态 h_0
        return torch.zeros(batch_size, self.hidden_dim).to(device)

    def forward_step(self, obs_input, prev_rnn_state):
        """
        单步前向传播
        obs_input: [B, State + Action] -> 当前时刻 t 的观测和动作
        prev_rnn_state: [B, Hidden] -> 上一时刻的确定性状态 h_t
        """
        # 1. 拆分输入
        state_t = obs_input[:, :self.state_dim]
        action_t = obs_input[:, self.state_dim:]

        # 2. Embedding
        s_embed = self.state_encoder(state_t)
        a_embed = self.action_encoder(action_t)

        # 3. Posterior Inference (q(z_t | h_t, s_t))
        # 利用 h_t (来自 t-1 的记忆) 和 s_t (当前观测)
        post_in = torch.cat([prev_rnn_state, s_embed], dim=-1)
        post_stats = self.posterior_net(post_in)
        post_mu, post_logvar = torch.chunk(post_stats, 2, dim=-1)
        
        # 4. Prior Inference (p(z_t | h_t))
        # 仅利用 h_t (用于计算 KL Loss)
        prior_stats = self.prior_net(prev_rnn_state)
        prior_mu, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)

        # 5. Sample z_t (Reparameterization)
        # 训练时使用 Posterior 采样
        z_t = self._sample_z(post_mu, post_logvar)

        # 6. Update RNN State (Deterministic Path)
        # h_{t+1} = GRU(h_t, z_t, a_t)
        # 注意：这里我们使用 action_t 来更新状态到 t+1
        rnn_in = torch.cat([z_t, a_embed], dim=-1)
        next_rnn_state = self.rnn_cell(rnn_in, prev_rnn_state)

        # 7. 打包返回
        # 我们需要返回 next_hidden 供下一次循环使用
        # 还需要返回 dist_params 供 compute_loss 使用
        # 这里的 z_t 将用于 Mixer (通过 sample_latents)
        
        # 将 z_t 和 next_rnn_state 打包在 dist_params 里传给 compute_loss
        # 注意：next_rnn_state 是 h_{t+1}, z_t 是 t 时刻的 latent
        return next_rnn_state, (prior_mu, prior_logvar, post_mu, post_logvar, z_t, next_rnn_state)

    def sample_latents(self, dist_params, num_samples=1):
        """
        返回给 Mixer 使用的特征。
        为了最大化信息量，我们返回 [h_t, z_t] 的拼接。
        注意：h_t 是 prev_rnn_state，z_t 是当前采样的。
        但是 forward_step 返回的 hidden 是 next_rnn_state (h_{t+1})。
        
        Mixer 需要的是 t 时刻的特征来辅助 Q(t)。
        z_t 是 t 时刻的。
        h_{t+1} 包含了 a_t，可能对 Q(t) 来说是未来的信息（如果我们认为 a_t 是 Q 的输出）。
        但通常 Q(s, a) 依赖 s。
        
        此处我们遵循 Dreamer 策略：Policy 输入是 [h_t, z_t]。
        但在 Learner 循环中，我们刚刚计算完 forward_step，手头有 z_t 和 h_{t+1} (next_rnn_state)。
        如果要获取 h_t，比较麻烦。
        
        **折衷方案**：Mixer 接收 [h_{t+1}, z_t]。
        理由：h_{t+1} 聚合了历史直到 t，包含了 z_t 和 a_t。这作为 Hypernet 输入是非常丰富的。
        """
        _, _, _, _, z_t, next_rnn_state = dist_params
        
        # Concat [h, z]
        # 维度变为: rssm_hidden_dim + rssm_latent_dim
        representation = torch.cat([next_rnn_state, z_t], dim=-1)
        
        # 增加维度适配 Learner [D, B, Latent] (D=1)
        return representation.unsqueeze(0)

    def compute_loss(self, z_samples, hidden_state, target_state, dist_params, target_reward=None):
        """
        z_samples: 从 sample_latents 返回的，这里我们不用，直接用 dist_params 里的
        hidden_state: h_{t+1}
        target_state: s_{t+1} (真实标签)
        dist_params: (p_mu, p_lv, q_mu, q_lv, z_t, h_next)
        """
        prior_mu, prior_logvar, post_mu, post_logvar, z_t, h_next = dist_params

        # 1. Image/State Reconstruction Loss (Predict S_{t+1})
        # 使用 h_{t+1} (包含了 s_t, a_t) 来预测 s_{t+1}
        # 为了增强随机性带来的鲁棒性，我们可以再次采样一个 z (来自 Prior) 拼接到 h_{t+1}，
        # 或者简单地把 z_t (Posterior) 和 h_next 拼起来。
        # 这里为了简单，且 z_t 已经是 latent，我们假设 h_next 足够预测。
        # 但标准的 decoder 输入通常是 [h, z]。
        # 让我们使用 [h_{t+1}, z_t] 作为解码输入，虽然 z_t 是 t 时刻的，但在 RNN 中这是常规操作。
        
        dec_in = torch.cat([h_next, z_t], dim=-1)
        pred_next_state = self.decoder(dec_in)
        
        recon_loss = F.mse_loss(pred_next_state, target_state, reduction='none').mean(dim=-1)

        # 2. KL Divergence Loss
        # KL(Posterior || Prior)
        # 限制后验分布不偏离先验太远
        kl_loss = self._kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)

        # 3. Reward Prediction Loss
        reward_loss = torch.zeros_like(recon_loss)
        if target_reward is not None:
            pred_reward = self.reward_head(dec_in)
            reward_loss = F.mse_loss(pred_reward, target_reward, reduction='none').squeeze(-1)

        return recon_loss, kl_loss, reward_loss

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu1, lv1, mu2, lv2):
        # KL(N(mu1, lv1) || N(mu2, lv2))
        var1 = torch.exp(lv1)
        var2 = torch.exp(lv2)
        kl = 0.5 * (var1 / var2 + (mu2 - mu1)**2 / var2 - 1 + lv2 - lv1)
        return kl.sum(dim=-1) # Sum over latent dim

    def predict(self, obs_input, hidden_state, use_mean=True):
        """
        用于评估时的单步预测
        """
        # RSSM 的 predict 需要维护 h 和 z
        # 这里为了适配 learner.evaluate_world_model 的简单接口
        # 我们执行一步 forward
        next_hidden, params = self.forward_step(obs_input, hidden_state)
        _, _, post_mu, post_logvar, z_t, h_next = params
        
        # 如果是评估，通常使用 Mean
        if use_mean:
            z = post_mu
        else:
            z = z_t
            
        dec_in = torch.cat([h_next, z], dim=-1)
        pred_next_state = self.decoder(dec_in)
        
        return pred_next_state, next_hidden