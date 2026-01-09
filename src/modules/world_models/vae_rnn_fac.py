# src/modules/world_models/vae_rnn_fac.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedVAERNN(nn.Module):
    """
    Factorized VAE-RNN World Model (因式分解 VAE-RNN 世界模型)
    
    设计目标:
    1. 解决同义反复 (Tautology): 输入仅使用局部 Obs，迫使模型推理未知信息，而非简单的 State 压缩。
    2. 解决纠缠 (Entanglement): 使用 Shared Encoder 独立处理每个 Agent，物理上隔离信息流。
    3. 去混淆 (Deconfounding): 通过 Attention 聚合局部动力学特征，生成 Proxy Confounder。
    4. 工程兼容: 支持 Dual-Run 模式（动态 Batch Size）。
    """

    def __init__(self, args, input_shape, output_shape):
        """
        Args:
            args: 全局参数配置对象。
            input_shape (int): Encoder 输入维度 = obs_dim + n_actions
            output_shape (int): Decoder 输出维度 = obs_dim (预测下一帧局部观测)
        """
        super(FactorizedVAERNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape = output_shape
        # [!!! 核心修复 !!!] 
        # 在这里实现解耦：
        # 如果 args 里有 wm_hidden_dim (128)，就用它作为 WM 的隐藏层。
        # 如果没有，才回退到 rnn_hidden_dim (64)。
        # 这样 Agent 用 64，WM 用 128，互不干扰。
        self.hidden_dim = getattr(args, "wm_hidden_dim", args.rnn_hidden_dim)

        
        # === 修复：兼容不同的参数命名 ===
        if hasattr(args, "wm_latent_dim"):
            self.latent_dim = args.wm_latent_dim
        elif hasattr(args, "latent_dim"):
            self.latent_dim = args.latent_dim
        
        # 聚合后的维度，建议与 Mixer 的 embedding 维度一致 (通常为 64 或 32)
        # 如果 args 中没有定义 mixing_embed_dim，默认为 64
        # self.att_embed_dim = getattr(args, "mixing_embed_dim", 64)
        self.att_embed_dim = self.latent_dim

        # ===================================================================
        # 1. Factorized Encoder (Shared Weights across Agents)
        #    核心思想：解决纠缠 (Entanglement)
        # ===================================================================
        # 我们将 Batch 和 N_Agents 维度合并处理，这意味着所有 Agent 共用这一套神经网络参数。
        # 物理含义：所有 Agent 遵循相同的物理定律，但它们的隐状态 z_i 是相互独立的。
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        
        # VAE 的均值和方差头
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # ===================================================================
        # 2. Factorized Decoder (Reconstruction)
        #    核心思想：监督信号来源
        # ===================================================================
        # 目标是重构局部观测 (Obs)，而不是全局 State。
        # 这迫使 z_local 捕获局部动力学特征（如：我是否在移动？敌人是否掉血？）。
        self.decoder_fc = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_out = nn.Linear(self.hidden_dim, output_shape)

        # ===================================================================
        # 3. Attention Aggregator (Proxy Confounder Generator)
        #    核心思想：去混淆 (Deconfounding)
        # ===================================================================
        # 作用：将 N 个独立的 z_i 聚合成全局 Z_global 给 Mixer。
        # 为什么用 Attention？因为它能捕获 Agent 间的动态交互关系（如集火、掩护）。
        # 这就是我们在论文中论证的 "Dynamics-based Proxy Confounder"。
        self.att_query = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_key = nn.Linear(self.latent_dim, self.att_embed_dim)
        self.att_val = nn.Linear(self.latent_dim, self.att_embed_dim)
        print("##########using new wm#########")

    def reparameterize(self, mu, logvar):
        """
        VAE 重参数化技巧 (Reparameterization Trick)
        
        优化点:
        - 训练时 (self.training=True): 采样 epsilon，允许梯度反传并增加模型鲁棒性。
        - 测试时 (self.training=False): 直接使用均值 mu，保证推理结果的确定性 (Deterministic)，减少方差。
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, inputs, actions, hidden_state=None):
        """
        前向传播逻辑
        
        Args:
            inputs (Tensor): [Batch, Seq, N_Agents, Obs_Dim] - 局部观测
            actions (Tensor): [Batch, Seq, N_Agents, Act_Dim] - 动作 (One-hot)
            hidden_state (Tensor): [1, Batch * N_Agents, Hidden_Dim] - RNN 记忆单元
                                   注意：这里的大小是 B*N，兼容 Dual Run 的动态 Batch。
        
        Returns:
            z_for_mixer: [Batch, Seq, Embed_Dim] -> 输送给 Mixer 的去混淆特征
            recon_out: [Batch, Seq, N_Agents, Obs_Dim] -> 用于计算 MSE Loss
            mu, logvar: -> 用于计算 KL Divergence Loss
            h_out: -> 下一步的 Hidden State
        """
        # 获取动态维度 (支持 Dual Run: B 可能是 WM_Batch 也可以是 RL_Batch)
        b, t, n, _ = inputs.shape
        
        # -------------------------------------------------------------------
        # Step A: 准备输入 (Solving Tautology)
        # -------------------------------------------------------------------
        # 严禁混入 Global State，只使用 Obs + Action。
        # 这样 Z 就不会是 State 的简单的有损压缩，而是对局部信息的时序推理。
        x = torch.cat([inputs, actions], dim=-1)  # [B, T, N, Obs_Dim + Act_Dim]
        
        # -------------------------------------------------------------------
        # Step B: 独立编码 (Solving Entanglement)
        # -------------------------------------------------------------------
        # [关键操作] 展平 Batch 和 Agent 维度: [B, T, N, ...] -> [B*N, T, ...]
        # 这使得随后的 FC 和 RNN 认为这是 (B*N) 个独立的序列。
        # 物理意义：切断了 Agent 之间的直接信息通路，实现因果隔离。
        x_flat = x.reshape(b * n, t, -1) 
        
        # 特征提取 MLP
        x_emb = F.relu(self.fc1(x_flat))
        
        # --- RNN 处理 (支持动态 Batch) ---
        if hidden_state is None:
            # 如果是序列开始或第一次调用，初始化全零 Hidden State
            # 使用 x_emb.new_zeros 确保设备(CPU/GPU)一致
            h_in = x_emb.new_zeros(1, b * n, self.hidden_dim)
        else:
            # 如果传入了 hidden_state，确保其形状匹配当前的 B*N
            # 这在 Dual Run 中非常重要，因为 batch size 会变
            h_in = hidden_state.reshape(1, b * n, -1)
            
        rnn_out, h_out = self.rnn(x_emb, h_in)
        
        # 计算潜在分布参数
        mu = self.fc_mu(rnn_out)       # [B*N, T, Latent_Dim]
        logvar = self.fc_logvar(rnn_out)
        
        # 采样得到局部隐变量 z_local
        z_local = self.reparameterize(mu, logvar)
        
        # -------------------------------------------------------------------
        # Step C: 局部重构 (Auxiliary Task / Training Signal)
        # -------------------------------------------------------------------
        # 尝试恢复局部观测，产生梯度信号
        recon_x = F.relu(self.decoder_fc(z_local))
        recon_out = self.decoder_out(recon_x) # [B*N, T, Obs_Dim]
        
        # -------------------------------------------------------------------
        # Step D: 恢复维度与聚合 (Aggregation for Deconfounding)
        # -------------------------------------------------------------------
        # 将扁平的维度恢复为 [B, T, N, ...]，准备进行 Agent 间的交互计算
        z_local = z_local.reshape(b, t, n, -1)
        recon_out = recon_out.reshape(b, t, n, -1)
        mu = mu.reshape(b, t, n, -1)
        logvar = logvar.reshape(b, t, n, -1)
        
        # --- Self-Attention Aggregation ---
        # 计算 Agent 间的动力学相关性，生成全局 Proxy Confounder
        q = self.att_query(z_local) # [B, T, N, Emb]
        k = self.att_key(z_local)   # [B, T, N, Emb]
        v = self.att_val(z_local)   # [B, T, N, Emb]
        
        # Scaled Dot-Product Attention
        # attention_score(i, j) 表示 Agent i 和 Agent j 在动力学上的关联程度
        scaling = self.att_embed_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scaling # [B, T, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权聚合: 此时 z_weighted 中的每个 Agent 特征都融合了与之相关的其他 Agent 信息
        z_attended = torch.matmul(attn_weights, v) # [B, T, N, Emb]


        # [修改]不要做 Mean Pooling 
        # 我们直接把每个 Agent 的 Z 给 Mixer，让 Mixer 自己决定怎么用
        z_for_mixer = z_attended  # 保持 [B, T, N, 64]
        
        return z_for_mixer, recon_out, mu, logvar, h_out
    
        # #------------------------------------------------------------------------------       
        # # --- Global Pooling ---
        # # 将 N 个 Agent 的特征压缩为一个全局向量，供 Mixer 使用
        # # Mean Pooling 是一种对 Agent 数量不敏感的聚合方式，利于迁移学习
        # z_for_mixer = z_weighted.mean(dim=2) # [B, T, Emb]
        
        # return z_for_mixer, recon_out, mu, logvar, h_out