import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HyperAttention(nn.Module):
    """
    [Context-Query Hypernetwork]
    Query: Agent Dynamic Roles (身份/职责)
    Key/Value: Global Context (S + Z)
    """
    def __init__(self, input_dim, embed_dim):
        super(HyperAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(input_dim)
        
        # 存储权重供 Learner 计算 Loss
        self.last_attn_weights = None

    def forward(self, role_embeddings, context):
        # context: [B, Dim] -> [B, 1, Dim]
        context = context.unsqueeze(1)
        
        q = self.q_proj(self.ln_q(role_embeddings)) # [B, N, E]
        k = self.k_proj(self.ln_k(context))         # [B, 1, E]
        v = self.v_proj(self.ln_k(context))         # [B, 1, E]
        
        scale = self.embed_dim ** -0.5
        attn_scores = th.matmul(q, k.transpose(-2, -1)) * scale
        
        # 使用 Tanh 允许负相关性 (即某种局势下，该角色被抑制)
        attn_weights = th.tanh(attn_scores) # [B, N, 1]
        self.last_attn_weights = attn_weights
        
        out = attn_weights * v 
        return self.out_proj(out + role_embeddings)

class DVDWMCausalMixer(nn.Module):
    def __init__(self, args):
        super(DVDWMCausalMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.latent_dim = args.wm_latent_dim 
        self.use_single = getattr(args, "use_single", False)

        # 1. [Identity] 角色原型 (Static ID)
        self.role_prototypes = nn.Parameter(th.randn(self.n_agents, self.embed_dim))
        
        # 2. [Dynamic] 角色调制器 (Z -> Modulation)
        self.role_modulator = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim),
            nn.ReLU()
        )

        # 3. [Interaction] HyperAttention
        self.context_dim = self.state_dim + self.latent_dim
        self.hyper_attention = HyperAttention(input_dim=self.context_dim, embed_dim=self.embed_dim)
        
        # Standard Bias & W2
        self.use_z_bias = getattr(args, "use_z_bias", False)
        self.bias_input_dim = self.latent_dim + self.state_dim if self.use_z_bias else self.state_dim
        self.hyper_b_1 = nn.Linear(self.bias_input_dim, self.embed_dim)

        self.hyper_hidden_dim = args.hypernet_hidden_dim
        w2_input_dim = self.state_dim if self.use_single else (self.state_dim + self.latent_dim)

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

    def forward(self, agent_qs, states, z_samples):
        D = z_samples.shape[0]
        bs, ts, _ = agent_qs.shape

        states_expanded = states.unsqueeze(0).expand(D, -1, -1, -1)
        qs_expanded = agent_qs.unsqueeze(0).expand(D, -1, -1, -1)
        
        states_reshaped = states_expanded.reshape(-1, self.state_dim)
        z_reshaped = z_samples.reshape(-1, self.latent_dim)
        qs_reshaped = qs_expanded.reshape(-1, 1, self.n_agents)
        
        # 1. Generate Dynamic Roles
        modulation = self.role_modulator(z_reshaped) 
        dynamic_roles = self.role_prototypes.unsqueeze(0) * modulation.unsqueeze(1) # [DBT, N, E]

        # 2. HyperAttention
        global_context = th.cat([states_reshaped, z_reshaped], dim=-1)
        w1_features = self.hyper_attention(dynamic_roles, global_context)
        w1 = th.abs(w1_features)
        
        # 3. Mix
        if self.use_z_bias:
            bias_input = th.cat([states_reshaped, z_reshaped], dim=-1)
        else:
            bias_input = states_reshaped
        b1 = self.hyper_b_1(bias_input).view(-1, 1, self.embed_dim)
        
        hidden = F.elu(th.bmm(qs_reshaped, w1) + b1)
        
        if self.use_single:
            w2_in = states_reshaped
        else:
            w2_in = global_context
            
        w2 = th.abs(self.hyper_w_2(w2_in)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(bias_input).view(-1, 1, 1)
        
        y = th.bmm(hidden, w2) + b2
        q_tot = y.view(D, bs, ts, 1).mean(dim=0)
        
        return q_tot