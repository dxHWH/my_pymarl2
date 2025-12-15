import torch
import torch.nn as nn
import torch.nn.functional as F

class VAERNNWorldModel(nn.Module):
    def __init__(self, input_dim, args):
        super(VAERNNWorldModel, self).__init__()
        self.args = args
        
        # 1. 维度解析
        # input_dim 是 Learner 传入的 (State_Dim + N_Agents * N_Actions)
        self.input_dim = input_dim 
        
        # 解析 State Dimension (Decoder 的预测目标维度)
        if isinstance(args.state_shape, int):
            self.state_dim = args.state_shape
        else:
            self.state_dim = int(args.state_shape[0])

        self.rnn_hidden_dim = args.wm_hidden_dim
        self.latent_dim = args.wm_latent_dim

        # 2. Encoder (Feature Extractor)
        # 输入: Current State + Joint Actions
        self.fc1 = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        
        # 3. RNN (Deterministic Path)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # 4. Posterior Encoder (Inference Path)
        # 从 RNN 隐状态推断 Z 的分布
        self.fc_mu = nn.Linear(self.rnn_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.rnn_hidden_dim, self.latent_dim)

        # 5. Decoder (Predictor)
        # 输入: z_t (随机) + h_t (记忆)
        # 输出: Next State (预测值) -> 维度应该是 state_dim !!!
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.rnn_hidden_dim, self.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.rnn_hidden_dim, self.state_dim) # <--- 修正: 输出维度为 state_dim
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.rnn_hidden_dim).to(device)

    def forward_step(self, obs_input, hidden_state):
        x = F.relu(self.fc1(obs_input))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        next_hidden = self.rnn(x, h_in)
        
        mu = self.fc_mu(next_hidden)
        logvar = self.fc_logvar(next_hidden)
        
        return next_hidden, (mu, logvar)

    def sample_latents(self, dist_params, num_samples=1):
        mu, logvar = dist_params
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, *std.shape).to(std.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0) 
        return z

    def compute_loss(self, z_samples, hidden_state, target_state, dist_params):
        """
        z_samples: [D, B, Latent]
        hidden_state: [B, Hidden]
        target_state: [B, State_Dim] (真实发生的 Next State)
        dist_params: (mu, logvar)
        """
        mu, logvar = dist_params
        
        # 重构 Loss (Reconstruction)
        # 简单起见，使用第一次采样的 z (或者均值 z) 来进行预测
        z_0 = z_samples[0] # [B, Latent]
        
        # 拼接 Input: [z, h]
        inp = torch.cat([z_0, hidden_state], dim=-1) # [B, Latent + Hidden]
        
        # 预测下一时刻的 State
        pred_next_state = self.decoder(inp) # [B, State_Dim]
        
        # 计算 MSE Loss
        # 确保 target_state 维度也是 [B, State_Dim]
        recon_loss = F.mse_loss(pred_next_state, target_state, reduction='none')
        recon_loss = recon_loss.mean(dim=-1) # Average over features
        
        # KL Divergence Loss
        # VAE 正则项: 限制分布接近 N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        return recon_loss, kl_loss