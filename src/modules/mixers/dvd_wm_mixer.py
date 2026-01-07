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
        
        # === 新增参数控制 ===
        self.use_facZ = getattr(args, "use_facZ", False) # 是否使用 Factorized WM 的 Z
        
        # 确定 Z 的维度
        if self.use_facZ:
            # Factorized 模式下，Z 是 Attention 聚合后的特征，维度由 mixing_embed_dim 决定
            # (参考 vae_rnn_fac.py 中的 att_embed_dim)
            self.latent_dim = args.mixing_embed_dim 
        else:
            # 旧模式下，Z 是 VAE 的隐变量
            self.latent_dim = args.wm_latent_dim 

        # 旧参数兼容
        self.use_multiple = getattr(args, "use_multiple", False)
        self.use_z_bias = getattr(args, "use_z_bias", False)
        # Factorized 模式下通常不使用 CMIX 的复杂乘法，而是直接拼接，所以强制检查
        if self.use_facZ:
             # 如果是 FacZ，建议使用拼接模式，暂时关闭 use_multiple 以免维度冲突
             # 或者你可以根据需要保留，但要注意维度匹配
             pass 

        # === 1. Hypernet 网络定义 ===
        
        # 定义 Hypernet 1 的输入维度
        if self.use_multiple:
             # CMIX 模式：W 只看 State
             self.hyper_input_dim = self.state_dim
        else:
             # 拼接模式：State + Z
             self.hyper_input_dim = self.state_dim + self.latent_dim

        # 构建 Hypernet 1
        if self.use_multiple:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )
            if self.latent_dim != self.n_agents:
                self.cid_map = nn.Linear(self.latent_dim, self.n_agents)
        else:
            # 标准拼接模式 (包括 FacZ 推荐模式)
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.hyper_input_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )

        # 定义 Bias 网络
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # 构建 Hypernet 2
        if self.use_multiple:
            self.hyper_w_2 = nn.Linear(self.state_dim, self.embed_dim)
        else:
            # 根据 use_single 参数决定 Layer 2 是否还看 Z
            # 如果 use_single=True, Layer 2 只看 State
            w2_input_dim = self.state_dim if getattr(self.args, "use_single", False) else self.hyper_input_dim
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(w2_input_dim, args.hypernet_hidden_dim),
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
        Args:
            agent_qs: [Batch, Seq, N_Agents]
            states: [Batch, Seq, State_Dim]
            z_samples: 
                - 如果 use_facZ=True: [Batch, Seq, Z_Dim] (无采样维度 D)
                - 如果 use_facZ=False: [D, Batch, Seq, Z_Dim] (可能有采样维度 D)
        """
        
        # === 维度处理与对齐 ===
        if self.use_facZ:
            # FacZ 模式：没有 D 维度，手动增加一个维度以兼容后续逻辑 D=1
            D = 1
            bs, ts, _ = agent_qs.shape
            # z_samples: [B, T, Z] -> [1, B, T, Z]
            z_samples = z_samples.unsqueeze(0) 
        else:
            # 旧模式：可能有 D 维度
            if z_samples.dim() == 4:
                D = z_samples.shape[0]
            else:
                D = 1
                z_samples = z_samples.unsqueeze(0)
            bs, ts, _ = agent_qs.shape

        # Reshape for Batch Processing
        # [D, B, T, State_Dim] -> [D*B*T, State_Dim]
        states_reshaped = states.unsqueeze(0).expand(D, -1, -1, -1).reshape(-1, self.state_dim)
        z_reshaped = z_samples.reshape(-1, self.latent_dim)
        qs_reshaped = agent_qs.unsqueeze(0).expand(D, -1, -1, -1).reshape(-1, 1, self.n_agents)

        # === Hypernet 1 计算 ===
        if self.use_multiple:
            # --- CMIX 乘法模式 ---
            w1_psi = th.abs(self.hyper_w_1(states_reshaped)).view(-1, self.n_agents, self.embed_dim)
            
            cid = z_reshaped
            if hasattr(self, "cid_map"):
                cid = self.cid_map(cid)
            
            # W = W_psi * CID
            w1 = w1_psi * cid.unsqueeze(-1)
        else:
            # --- 拼接模式 (FacZ 默认走这里) ---
            hyper_input = th.cat([states_reshaped, z_reshaped], dim=-1)
            w1 = th.abs(self.hyper_w_1(hyper_input)).view(-1, self.n_agents, self.embed_dim)

        # === Bias 1 计算 ===
        bias_input = th.cat([states_reshaped, z_reshaped], dim=-1) if self.use_z_bias else states_reshaped
        b1 = self.hyper_b_1(bias_input).view(-1, 1, self.embed_dim)
        
        # Layer 1 输出
        hidden = F.elu(th.bmm(qs_reshaped, w1) + b1)

        # === Hypernet 2 计算 ===
        if self.use_multiple:
            w2 = th.abs(self.hyper_w_2(states_reshaped)).view(-1, self.embed_dim, 1)
        else:
            w2_input = states_reshaped if getattr(self.args, "use_single", False) else th.cat([states_reshaped, z_reshaped], dim=-1)
            w2 = th.abs(self.hyper_w_2(w2_input)).view(-1, self.embed_dim, 1)
        
        b2 = self.hyper_b_2(bias_input).view(-1, 1, 1)
        
        # Layer 2 输出
        y = th.bmm(hidden, w2) + b2
        
        # 恢复维度 [D, B, T, 1] -> 平均 D -> [B, T, 1]
        return y.view(D, bs, ts, 1).mean(dim=0)