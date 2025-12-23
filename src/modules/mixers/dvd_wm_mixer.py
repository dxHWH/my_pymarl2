import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DVDWMMixer(nn.Module):
    def __init__(self, args):
        super(DVDWMMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.latent_dim = args.wm_latent_dim # 使用新的 wm 参数名
        self.use_single = args.use_single #指定多少层z的参与


        # [Trick 1:] Confounder Bias Network
        # 专门用于拟合由潜在变量 z 引起的全局价值偏差
        self.use_z_bias = getattr(args, "use_z_bias", False)
        # 如果启用 bias，输入为 State + Z；否则仅为 State
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim

        # Hypernet 1: State + Global_Z -> W1
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, args.hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
        )
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # Hypernet 2: State + Global_Z -> W2
        if self.use_single:
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )
            self.hyper_b_2 = nn.Sequential(
                nn.Linear(self.state_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, 1)
            )
        else:
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim + self.latent_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )
            self.hyper_b_2 = nn.Sequential(
                nn.Linear(self.bias_input_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, 1)
            )


     

    def forward(self, agent_qs, states, z_samples):
        """
        z_samples: [D, B, T, latent_dim] (已经是 Global Z)
        """
        # z_samples: [D, B, T, L]
        # states: [B, T, S]
        # agent_qs: [B, T, N]

        D = z_samples.shape[0]
        bs, ts, _ = agent_qs.shape

        # Expand dims for D samples
        states_expanded = states.unsqueeze(0).expand(D, -1, -1, -1) # [D, B, T, S]
        qs_expanded = agent_qs.unsqueeze(0).expand(D, -1, -1, -1)   # [D, B, T, N]
        
        # Reshape for Linear Layers
        states_reshaped = states_expanded.reshape(-1, self.state_dim)
        z_reshaped = z_samples.reshape(-1, self.latent_dim)
        qs_reshaped = qs_expanded.reshape(-1, 1, self.n_agents)
        
        # Concat State + Global Z
        hyper_input = th.cat([states_reshaped, z_reshaped], dim=-1)

        # === 修复点: 根据配置选择 Bias 网络的输入 ===
        if self.use_z_bias:
            bias_input = th.cat([states_reshaped, z_reshaped], dim=-1)
        else:
            bias_input = states_reshaped
        
        # --- QMIX Logic ---
        
        w1 = th.abs(self.hyper_w_1(hyper_input)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(bias_input).view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(qs_reshaped, w1) + b1)

        if self.use_single:
            w2 = th.abs(self.hyper_w_2(states_reshaped)).view(-1, self.embed_dim, 1)
            b2 = self.hyper_b_2(states_reshaped).view(-1, 1, 1)
        else:
            w2 = th.abs(self.hyper_w_2(hyper_input)).view(-1, self.embed_dim, 1)
            b2 = self.hyper_b_2(bias_input).view(-1, 1, 1)
        
       
        y = th.bmm(hidden, w2) + b2
        
        # Deconfounding (Mean over D)
        q_tot = y.view(D, bs, ts, 1).mean(dim=0)
        return q_tot