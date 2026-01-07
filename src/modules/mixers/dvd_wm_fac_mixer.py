import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DVDWMFacMixer(nn.Module):
    def __init__(self, args):
        """
        DVDWMFacMixer: 专为 Factorized World Model 设计的 Mixer。
        核心亮点: 支持 CMIX 风格的乘法调制机制 (Multiplicative Modulation)。
        """
        super(DVDWMFacMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        
        # FacWM 输出的 Z 维度通常与 mixing_embed_dim 一致 (e.g. 64)
        # 如果 args 中有专门定义的 fac_z_dim 则使用，否则复用 mixing_embed_dim
        self.latent_dim = getattr(args, "wm_latent_dim", 64) 
        if getattr(args, "use_facZ", False):
             self.latent_dim = args.mixing_embed_dim

        # === 核心开关 ===
        # use_multiple: 是否使用乘法调制 (True=CMIX风格, False=拼接风格)
        self.use_multiple = getattr(args, "use_multiple", False)
        # use_z_bias: Bias 网络是否也需要看 Z (通常建议 True)
        self.use_z_bias = getattr(args, "use_z_bias", True)

        # =======================================================================
        # Hypernet 1: 生成第一层权重 W1 [N_Agents -> Embed_Dim]
        # =======================================================================
        if self.use_multiple:
            # --- 乘法模式 (Multiplicative) ---
            # 1. Base Weights: 仅由 State 生成
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )
            
            # 2. Modulator Map: 将 Z 映射到 Agent 维度
            # Z (64) -> CID (N_Agents)
            # 这决定了每个 Agent 的权重被放大还是缩小
            self.cid_map = nn.Linear(self.latent_dim, self.n_agents)
            
        else:
            # --- 拼接模式 (Concatenative) ---
            # W1 由 State + Z 共同生成
            self.hyper_input_dim = self.state_dim + self.latent_dim
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.hyper_input_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )

        # =======================================================================
        # Hypernet Bias 1: 生成第一层偏置 b1 [Embed_Dim]
        # =======================================================================
        # Bias 决定了 Value 的基准值，建议始终让它看到 Z
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # =======================================================================
        # Hypernet 2: 生成第二层权重 W2 [Embed_Dim -> 1]
        # =======================================================================
        if self.use_multiple:
            # CMIX 风格通常在第二层只使用 State，或者保持简单
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )
        else:
            # 拼接模式
            w2_input_dim = self.state_dim if getattr(self.args, "use_single", False) else self.state_dim + self.latent_dim
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(w2_input_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim)
            )

        # =======================================================================
        # Hypernet Bias 2: 生成最终偏置 b2 [1] (即 V(s))
        # =======================================================================
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.bias_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, z):
        """
        Args:
            agent_qs: [Batch, Seq, N_Agents]
            states: [Batch, Seq, State_Dim]
            z: [Batch, Seq, Z_Dim] (来自 FacWM 的聚合动力学特征)
        """
        bs, ts, _ = agent_qs.shape
        
        # 展平 Batch 和 Seq 维度
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        z = z.reshape(-1, self.latent_dim)

        # ===================================================================
        # Layer 1 权重计算 (W1)
        # ===================================================================
        if self.use_multiple:
            # --- 乘法调制逻辑 ---
            # 1. 计算基础权重 (State-dependent)
            w1_base = th.abs(self.hyper_w_1(states)) # [B*T, N * Embed]
            w1_base = w1_base.view(-1, self.n_agents, self.embed_dim)
            
            # 2. 计算调节因子 (Dynamics-dependent)
            # cid: [B*T, Z] -> [B*T, N]
            # 这里不需要 Abs 或 ReLU，允许 Z 对权重进行正负调节? 
            # 通常 QMIX 要求权重非负。WM 的 Z 可能会有负值。
            # 建议对 cid 进行 abs 或 sigmoid 处理，或者只对 w1_base 取 abs，cid 保持线性但最终结果取 abs
            # QMIX 约束: weights must be non-negative.
            # 做法 A: abs(Base) * abs(CID)
            # 做法 B: abs(Base * CID) -> 我们采用这个，更灵活
            
            cid = self.cid_map(z) 
            
            # [B*T, N, 1] * [B*T, N, Embed] (Broadcasting)
            # Z 决定了哪些 Agent 的权重被增强或抑制
            w1 = th.abs(w1_base * cid.unsqueeze(-1))
            
        else:
            # --- 拼接逻辑 ---
            inputs = th.cat([states, z], dim=1)
            w1 = th.abs(self.hyper_w_1(inputs))
            w1 = w1.view(-1, self.n_agents, self.embed_dim)

        # ===================================================================
        # Layer 1 计算
        # ===================================================================
        # Bias
        bias_in = th.cat([states, z], dim=1) if self.use_z_bias else states
        b1 = self.hyper_b_1(bias_in).view(-1, 1, self.embed_dim)
        
        # Hidden = (Q * W1) + b1
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # ===================================================================
        # Layer 2 权重计算 (W2)
        # ===================================================================
        if self.use_multiple:
            w2 = th.abs(self.hyper_w_2(states)).view(-1, self.embed_dim, 1)
        else:
            w2_in = states if getattr(self.args, "use_single", False) else th.cat([states, z], dim=1)
            w2 = th.abs(self.hyper_w_2(w2_in)).view(-1, self.embed_dim, 1)

        # ===================================================================
        # Layer 2 计算
        # ===================================================================
        b2 = self.hyper_b_2(bias_in).view(-1, 1, 1)
        
        # Q_tot = (Hidden * W2) + b2
        y = th.bmm(hidden, w2) + b2
        
        return y.view(bs, ts, 1)