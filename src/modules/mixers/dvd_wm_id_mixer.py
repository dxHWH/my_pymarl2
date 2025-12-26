import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HyperAttention(nn.Module):
    """
    [Context-Query Hypernetwork Module]
    实现了从 "被动广播" 到 "主动检索" 的范式转变。
    
    Query: Agent Dynamic Roles ("我是谁/我现在的职责")
    Key/Value: Global Context (State + Z) ("现在的战局/环境")
    
    逻辑: 每个 Agent 根据自己的动态角色，主动从全局上下文中检索与自己归因相关的特征。
    """
    def __init__(self, input_dim, embed_dim):
        super(HyperAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. 投影层
        # Query 投影: 处理动态角色
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Key/Value 投影: 处理全局上下文
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        
        # 2. 输出层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # LayerNorm 保证数值稳定性
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(input_dim)

    def forward(self, role_embeddings, context):
        """
        Args:
            role_embeddings (Query): [Batch, N_Agents, Embed_Dim]
            context (Key/Value):     [Batch, Context_Dim]
        Returns:
            w1_weights: [Batch, N_Agents, Embed_Dim]
        """
        # context 需要增加序列维度以便 Attention 处理: [Batch, Context_Dim] -> [Batch, 1, Context_Dim]
        context = context.unsqueeze(1)
        
        # 1. Projections
        # Q = Role
        q = self.q_proj(self.ln_q(role_embeddings)) # [B, N, E]
        
        # K, V = Context (Context 只有一个 token，所以 seq_len=1)
        k_input = self.ln_k(context)
        k = self.k_proj(k_input) # [B, 1, E]
        v = self.v_proj(k_input) # [B, 1, E]
        
        # 2. Scaled Dot-Product Attention
        # Scores: Q * K^T
        # [B, N, E] @ [B, E, 1] -> [B, N, 1]
        # 物理意义: 计算每个 Agent 的 Role 与当前局势(Context)的"匹配度"或"相关性"
        scale = self.embed_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 3. Attention Weights
        # 既然 Key 只有一个 (Global Context)，Softmax 会导致全是 1。
        # 这里我们需要的是一种"门控"或"强度"机制，而非概率分布。
        # Tanh 允许负相关性 (即某种局势下，该角色权重被抑制)，Sigmoid 则仅允许正向门控。
        # 这里使用 Tanh 以获得更丰富的非线性交互。
        attn_weights = torch.tanh(attn_scores) # [B, N, 1]
        
        # 4. Aggregation (Value Retrieval)
        # Context Features 被每个 Agent 根据相关性"截取"
        # [B, N, 1] * [B, 1, E] -> [B, N, E] (Broadcasting)
        context_aware_features = attn_weights * v 
        
        # 5. Residual Connection
        # 把原始的角色信息加回来，防止丢失身份底色
        out = context_aware_features + role_embeddings
        
        return self.out_proj(out)


class DVDWMMixer(nn.Module):
    def __init__(self, args):
        super(DVDWMMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.latent_dim = args.wm_latent_dim 
        self.use_single = getattr(args, "use_single", False)

        # -----------------------------------------------------------
        # [Mechanism 1]: Dynamic Role Generation (动态角色生成)
        # -----------------------------------------------------------
        # 1. 角色原型 (Role Prototypes): Agent 的性格底色 (Static)
        self.role_prototypes = nn.Parameter(th.randn(self.n_agents, self.embed_dim))
        
        # 2. 角色调制器 (Role Modulator): 根据战局 Z 调整角色 (Dynamic)
        # Z -> Modulation Vector
        self.role_modulator = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim),
            nn.ReLU()
        )

        # -----------------------------------------------------------
        # [Mechanism 2]: HyperAttention (注意力混合机制)
        # -----------------------------------------------------------
        # 上下文维度 = 全局状态 + 世界模型潜变量
        self.context_dim = self.state_dim + self.latent_dim
        
        # 替代了传统的 agent_weight_net MLP
        self.hyper_attention = HyperAttention(input_dim=self.context_dim, embed_dim=self.embed_dim)
        
        # -----------------------------------------------------------
        # Standard QMIX Hypernetworks (Bias & W2)
        # -----------------------------------------------------------
        # Hypernet Bias 1 (B1)
        self.use_z_bias = getattr(args, "use_z_bias", False)
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        # Hypernet 2 (W2, B2)
        # W2 主要负责全局加权，保持 MLP 结构即可
        self.hyper_hidden_dim = args.hypernet_hidden_dim
        if self.use_single:
            w2_input_dim = self.state_dim
        else:
            w2_input_dim = self.state_dim + self.latent_dim

        self.hyper_w_2 = nn.Sequential(
            nn.Linear(w2_input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim)
        )
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.bias_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        # 用于存储中间变量供 Learner 计算正则化 Loss
        self.last_dynamic_roles = None

    def forward(self, agent_qs, states, z_samples):
        """
        Args:
            agent_qs: [B, T, N]
            states:   [B, T, S]
            z_samples: [D, B, T, L]
        """
        D = z_samples.shape[0]
        bs, ts, _ = agent_qs.shape

        # ---------------------------------------------------
        # 1. Data Prep & Reshape (处理多样本 D)
        # ---------------------------------------------------
        states_expanded = states.unsqueeze(0).expand(D, -1, -1, -1) # [D, B, T, S]
        qs_expanded = agent_qs.unsqueeze(0).expand(D, -1, -1, -1)   # [D, B, T, N]
        
        # Flatten [D, B, T] -> [Total_Batch]
        states_reshaped = states_expanded.reshape(-1, self.state_dim)     # [DBT, S]
        z_reshaped = z_samples.reshape(-1, self.latent_dim)               # [DBT, L]
        qs_reshaped = qs_expanded.reshape(-1, 1, self.n_agents)           # [DBT, 1, N]
        
        # ---------------------------------------------------
        # 2. Dynamic Role Generation (Z -> Role)
        # ---------------------------------------------------
        # Z 是战局的压缩表征，它决定了当下的 Role
        modulation = self.role_modulator(z_reshaped) # [DBT, Embed]
        
        # Dynamic Role = Prototype * Modulation
        # [1, N, E] * [DBT, 1, E] -> [DBT, N, E] (Broadcasting)
        # 物理意义: 原型决定了Agent的基础特性，Modulation 决定了在当前 Z 下这些特性被激活/抑制的程度
        dynamic_roles = self.role_prototypes.unsqueeze(0) * modulation.unsqueeze(1)
        
        # [Critical]: 保存供 Learner 计算 "Attribution Consistency Loss"
        # 恢复维度 [D, B, T, N, E] -> 取 mean 或者直接保存 reshape 后的引用
        # 这里为了 Learner 方便，我们保存 View 之后的 Tensor
        # Learner 那边通常只需要 [B, T, N, E] (均值)
        self.last_dynamic_roles = dynamic_roles.view(D, bs, ts, self.n_agents, -1).mean(dim=0)

        # ---------------------------------------------------
        # 3. HyperAttention (Generate W1)
        # ---------------------------------------------------
        # 准备 Global Context (Key/Value)
        global_context = th.cat([states_reshaped, z_reshaped], dim=-1) # [DBT, S+L]
        
        # Attention Forward
        # Query=Dynamic Roles, Key/Val=Global Context
        w1_features = self.hyper_attention(dynamic_roles, global_context) # [DBT, N, Embed]
        
        # QMIX 约束: 权重必须为非负
        w1 = th.abs(w1_features)
        
        # ---------------------------------------------------
        # 4. Standard QMIX Computation
        # ---------------------------------------------------
        # B1 Generation
        if self.use_z_bias:
            bias_input = th.cat([states_reshaped, z_reshaped], dim=-1)
        else:
            bias_input = states_reshaped
        b1 = self.hyper_b_1(bias_input).view(-1, 1, self.embed_dim)
        
        # Layer 1: Q_hidden = ELU( Q * W1 + B1 )
        # [DBT, 1, N] * [DBT, N, E] -> [DBT, 1, E]
        hidden = F.elu(th.bmm(qs_reshaped, w1) + b1)
        
        # W2 Generation
        if self.use_single:
            w2_in = states_reshaped
        else:
            w2_in = global_context
            
        w2 = th.abs(self.hyper_w_2(w2_in)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(bias_input).view(-1, 1, 1)
        
        # Layer 2: Q_tot = Hidden * W2 + B2
        y = th.bmm(hidden, w2) + b2
        
        # ---------------------------------------------------
        # 5. Deconfounding Aggregation
        # ---------------------------------------------------
        # Reshape back and Mean over samples D
        q_tot = y.view(D, bs, ts, 1).mean(dim=0)
        
        return q_tot