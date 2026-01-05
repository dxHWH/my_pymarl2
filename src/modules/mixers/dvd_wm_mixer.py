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
        self.latent_dim = args.wm_latent_dim 
        
        # 新增超参：是否使用 CMIX 风格的乘法调制
        self.use_multiple = getattr(args, "use_multiple", False)
        self.use_z_bias = getattr(args, "use_z_bias", False)

        # === 1. 网络定义：兼容两种模式 ===
        if self.use_multiple:
            # CMIX 模式：基础权重网络 W_psi 只输入 State [cite: 233]
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )
            # CMIX 模式下，CID (z) 可能还需要一个线性层映射到 agent 数量 (如果 latent_dim != n_agents)
            if self.latent_dim != self.n_agents:
                self.cid_map = nn.Linear(self.latent_dim, self.n_agents)
        else:
            # 当前模式：State + Z 拼接输入
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim + self.latent_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )

        # Bias 输入维度逻辑保持兼容
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # Hypernet 2 逻辑
        if self.use_multiple:
            self.hyper_w_2 = nn.Linear(self.state_dim, self.embed_dim)
        else:
            input_dim = self.state_dim if self.args.use_single else self.state_dim + self.latent_dim
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(input_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )
        
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.bias_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, z_samples):
        D = z_samples.shape[0]
        bs, ts, _ = agent_qs.shape

        states_reshaped = states.unsqueeze(0).expand(D, -1, -1, -1).reshape(-1, self.state_dim)
        z_reshaped = z_samples.reshape(-1, self.latent_dim)
        qs_reshaped = agent_qs.unsqueeze(0).expand(D, -1, -1, -1).reshape(-1, 1, self.n_agents)

        # === 核心逻辑：根据 use_multiple 分支 ===
        
        if self.use_multiple:
            # --- CMIX 乘法模式 ---
            # 1. 计算基础权重 W_psi [cite: 231, 233]
            w1_psi = th.abs(self.hyper_w_1(states_reshaped)).view(-1, self.n_agents, self.embed_dim)
            
            # 2. 处理 CID (z) 并进行修正 [cite: 231, 235]
            cid = z_reshaped
            if hasattr(self, "cid_map"):
                cid = self.cid_map(cid) # 确保维度是 [batch, n_agents]
            
            # W = W_psi * CID_w (CID 作为修正算子相乘) [cite: 231, 233]
            w1 = w1_psi * cid.unsqueeze(-1) 
        else:
            # --- 当前拼接模式 ---
            hyper_input = th.cat([states_reshaped, z_reshaped], dim=-1)
            w1 = th.abs(self.hyper_w_1(hyper_input)).view(-1, self.n_agents, self.embed_dim)

        # Bias 逻辑保持不变
        bias_input = th.cat([states_reshaped, z_reshaped], dim=-1) if self.use_z_bias else states_reshaped
        b1 = self.hyper_b_1(bias_input).view(-1, 1, self.embed_dim)
        
        # 混合 Layer 1
        hidden = F.elu(th.bmm(qs_reshaped, w1) + b1)

        # Layer 2 逻辑
        if self.use_multiple:
            w2 = th.abs(self.hyper_w_2(states_reshaped)).view(-1, self.embed_dim, 1)
        else:
            w2_input = states_reshaped if self.args.use_single else th.cat([states_reshaped, z_reshaped], dim=-1)
            w2 = th.abs(self.hyper_w_2(w2_input)).view(-1, self.embed_dim, 1)
        
        b2 = self.hyper_b_2(bias_input).view(-1, 1, 1)
        
        y = th.bmm(hidden, w2) + b2
        return y.view(D, bs, ts, 1).mean(dim=0)