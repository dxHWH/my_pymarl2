import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DVDWMFacMixer(nn.Module):
    def __init__(self, args):
        """
        DVDWMFacMixer: 专为 Factorized World Model 设计的 Mixer。
        核心升级: 支持 Agent-wise 的乘法调制机制 (Fine-grained Deconfounding)。
        """
        super(DVDWMFacMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        print("#######using new mixer#######")
        
        # === Z 维度对齐 ===
        # 必须与 vae_rnn_fac.py 的输出维度一致 (即 wm_latent_dim)
        # 即使 args 中没有定义，默认给 64 以防报错
        if hasattr(args, "wm_latent_dim"):
            self.latent_dim = args.wm_latent_dim
        else:
            self.latent_dim = getattr(args, "latent_dim", 64)

        # === 核心开关 ===
        # use_multiple: 建议设为 True 以启用 Agent-wise 调制
        self.use_multiple = getattr(args, "use_multiple", False)
        # use_z_bias: Bias 网络是否也需要看 Z (通常建议 True)
        self.use_z_bias = getattr(args, "use_z_bias", True)

        # =======================================================================
        # Hypernet 1: 生成第一层权重 W1 [N_Agents -> Embed_Dim]
        # =======================================================================
        if self.use_multiple:
            # --- 乘法模式 (Multiplicative / Agent-wise) ---
            # 1. Base Weights: 仅由 State 生成 (捕捉全局局势)
            # 输出: [Batch, N_Agents * Embed_Dim] -> View 为 [Batch, N_Agents, Embed_Dim]
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )
            
            # 2. [关键升级] Agent-wise Modulator
            # 输入: 每个 Agent 自己的 z_i (Latent_Dim)
            # 输出: 该 Agent 对应的权重缩放因子 (Embed_Dim)
            # 这是一个 Parameter-Shared MLP，对 N 个 Agent 独立作用
            self.z_modulator = nn.Sequential(
                nn.Linear(self.latent_dim, 32), # 中间层维度，32 是一个经验值
                nn.ReLU(),
                nn.Linear(32, self.embed_dim)   # 输出维度与 Embedding 对齐
            )
            
        else:
            # --- 拼接模式 (Concatenative / Legacy) ---
            # 为了兼容性保留。如果 Z 是 Agent-wise 的，这里会对其做 Mean Pooling 变回 Global Z
            self.hyper_input_dim = self.state_dim + self.latent_dim
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.hyper_input_dim, args.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hypernet_hidden_dim, self.embed_dim * self.n_agents)
            )

        # =======================================================================
        # Hypernet Bias 1: 生成第一层偏置 b1 [Embed_Dim]
        # =======================================================================
        # Bias 是全局的，所以输入是 State + Global_Z
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
            states:   [Batch, Seq, State_Dim]
            z:        [Batch, Seq, N_Agents, Latent_Dim] (Agent-wise Z)
        """
        bs, ts, _ = agent_qs.shape
        
        # 展平 Batch 和 Seq 维度
        states = states.reshape(-1, self.state_dim)           # [B*T, State_Dim]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)     # [B*T, 1, N]
        
        # 处理 Z: 展平为 [B*T, N, Z_dim]
        # 注意: 即使 WM 输出是 Global 的 (N=1)，这里 reshape 也能兼容广播
        z = z.reshape(-1, self.n_agents, self.latent_dim)

        # 准备全局 Z (Global Z) 用于 Bias 计算
        # 对 Agent 维度求平均: [B*T, N, Z] -> [B*T, Z]
        z_global = z.mean(dim=1)

        # ===================================================================
        # Layer 1 权重计算 (W1)
        # ===================================================================
        if self.use_multiple:
            # --- 乘法调制逻辑 (Agent-wise) ---
            
            # 1. 计算基础权重 (State-dependent)
            # w1_base: [B*T, N * Embed] -> [B*T, N, Embed]
            w1_base = th.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
            
            # 2. 计算调节因子 (Dynamics-dependent)
            # 输入: z [B*T, N, Z_dim]
            # 输出: z_mod [B*T, N, Embed_Dim]
            # 这里的 MLP 是 Parameter Shared 的，对每个 Agent 独立运算
            z_mod = self.z_modulator(z)
            
            # 3. 元素级乘法调制
            # w1: [B*T, N, Embed]
            # 每个 Agent 的 Embedding 权重被其自身的动力学状态 z_i 缩放
            w1 = th.abs(w1_base * z_mod)
            
        else:
            # --- 拼接逻辑 (Legacy) ---
            # 使用全局 Z 进行拼接
            inputs = th.cat([states, z_global], dim=1)
            w1 = th.abs(self.hyper_w_1(inputs))
            w1 = w1.view(-1, self.n_agents, self.embed_dim)

        # ===================================================================
        # Layer 1 计算
        # ===================================================================
        # Bias 使用全局信息
        bias_in = th.cat([states, z_global], dim=1) if self.use_z_bias else states
        b1 = self.hyper_b_1(bias_in).view(-1, 1, self.embed_dim)
        
        # Hidden = (Q * W1) + b1
        # [B*T, 1, N] @ [B*T, N, Embed] -> [B*T, 1, Embed]
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # ===================================================================
        # Layer 2 权重计算 (W2)
        # ===================================================================
        if self.use_multiple:
            w2 = th.abs(self.hyper_w_2(states)).view(-1, self.embed_dim, 1)
        else:
            w2_in = states if getattr(self.args, "use_single", False) else th.cat([states, z_global], dim=1)
            w2 = th.abs(self.hyper_w_2(w2_in)).view(-1, self.embed_dim, 1)

        # ===================================================================
        # Layer 2 计算
        # ===================================================================
        b2 = self.hyper_b_2(bias_in).view(-1, 1, 1)
        
        # Q_tot = (Hidden * W2) + b2
        y = th.bmm(hidden, w2) + b2
        
        return y.view(bs, ts, 1)