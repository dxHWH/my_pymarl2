import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EntityAttention(nn.Module):
    """
    [World Model 的注意力编码器]
    用于识别哪些 Agent 的动作对环境动力学(Next State)最关键。
    Query: State (环境现状)
    Key: Agent Actions (个体行为)
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(EntityAttention, self).__init__()
        
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        
        # Attention Components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5

    def forward(self, state, joint_actions):
        """
        state: [B, S_dim]
        joint_actions: [B, N_Agents, A_dim]
        """
        # 1. Embeddings
        s_emb = self.state_embed(state).unsqueeze(1) # [B, 1, H]
        a_emb = self.action_embed(joint_actions)     # [B, N, H]
        
        # 2. Q, K, V
        q = self.query(s_emb) # [B, 1, H]
        k = self.key(a_emb)   # [B, N, H]
        v = self.value(a_emb) # [B, N, H]
        
        # 3. Attention Scores
        # Q * K^T -> [B, 1, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 4. Attention Weights (Softmax 归一化，代表物理重要性分布)
        weights = torch.softmax(scores, dim=-1) # [B, 1, N]
        
        # 5. Weighted Sum
        out = torch.matmul(weights, v).squeeze(1) # [B, H]
        
        # Residual connection with state
        final_out = out + s_emb.squeeze(1)
        
        return final_out, weights.squeeze(1) # [B, H], [B, N]


class VAERNNAttentionWorldModel(nn.Module):
    def __init__(self, input_dim, args):
        super(VAERNNAttentionWorldModel, self).__init__()
        self.args = args
        self.state_dim = int(args.state_shape)
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.latent_dim = args.wm_latent_dim

        # [修改] 使用 Attention Encoder 替代原来的 fc1
        # 原来的 input_dim 是 state + joint_actions 的总维数
        # 这里我们显式分开处理
        self.encoder = EntityAttention(self.state_dim, self.n_actions, self.rnn_hidden_dim)

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        
        self.fc_mu = nn.Linear(self.rnn_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.rnn_hidden_dim, self.latent_dim)
        
        # Decoder (P(S'|z, h))
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.rnn_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim + 1) # +1 for reward
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)

    def forward_step(self, state, hidden_state, joint_actions):
        """
        [修改] 显式接收 joint_actions [B, N, A] 以便计算 Attention
        """
        # 1. Encode (with Attention)
        # x: [B, H], attn_weights: [B, N]
        x, attn_weights = self.encoder(state, joint_actions)
        x = F.relu(x)
        
        # 2. RNN Step
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        next_hidden = self.rnn(x, h_in)
        
        # 3. Latent Dist
        mu = self.fc_mu(next_hidden)
        logvar = self.fc_logvar(next_hidden)
        
        return next_hidden, (mu, logvar), attn_weights

    def sample_latents(self, dist_params, num_samples=1):
        mu, logvar = dist_params
        std = torch.exp(0.5 * logvar)
        
        # [num_samples, B, latent_dim]
        eps = torch.randn(num_samples, mu.size(0), mu.size(1)).to(mu.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z

    def compute_loss(self, z_samples, hidden_state, target_state, dist_params, target_reward):
        """
        计算 ELBO Loss
        """
        mu, logvar = dist_params
        D, B, L = z_samples.shape
        
        # Expand hidden state to match samples
        hidden_expanded = hidden_state.unsqueeze(0).expand(D, -1, -1) # [D, B, H]
        
        # Decoder Input: Cat(z, h)
        decoder_input = torch.cat([z_samples, hidden_expanded], dim=-1) # [D, B, L+H]
        
        # Predict
        pred_out = self.fc_decoder(decoder_input) # [D, B, S+1]
        pred_state = pred_out[:, :, :-1]
        pred_reward = pred_out[:, :, -1:]
        
        # Expand Targets
        target_state_exp = target_state.unsqueeze(0).expand(D, -1, -1)
        target_reward_exp = target_reward.unsqueeze(0).expand(D, -1, -1)
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = ((pred_state - target_state_exp) ** 2).mean(0).sum(-1) # Mean over D, Sum over features
        
        # 2. Reward Loss (MSE)
        reward_loss = ((pred_reward - target_reward_exp) ** 2).mean(0).sum(-1)
        
        # 3. KL Divergence
        # Analytical KL for VAE
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        return recon_loss, kl_loss, reward_loss
    
    def predict(self, state, hidden_state, joint_actions, use_mean=True):
        # Evaluation function
        x, _ = self.encoder(state, joint_actions) # Ignore weights during eval
        x = F.relu(x)
        next_hidden = self.rnn(x, hidden_state)
        mu = self.fc_mu(next_hidden)
        
        z = mu # Deterministic for prediction
        decoder_input = torch.cat([z, next_hidden], dim=-1)
        pred_out = self.fc_decoder(decoder_input)
        return pred_out[:, :-1], next_hidden