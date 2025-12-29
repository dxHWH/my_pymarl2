import torch
import torch.nn as nn
import torch.nn.functional as F

class VaeRNN(nn.Module):
    def __init__(self, input_dim, args):
        super(VaeRNN, self).__init__()
        self.args = args
        
        # 1. 维度解析
        self.input_dim = input_dim 
        
        # 解析 State Dimension
        if isinstance(args.state_shape, int):
            self.state_dim = args.state_shape
        else:
            self.state_dim = int(args.state_shape[0])

        # 支持参数名兼容 (wm_hidden_dim 或 rnn_hidden_dim)
        self.rnn_hidden_dim = getattr(args, "wm_hidden_dim", getattr(args, "rnn_hidden_dim", 64))
        self.latent_dim = getattr(args, "wm_latent_dim", getattr(args, "latent_dim", 32))

        # 2. Encoder (Feature Extractor)
        self.fc_input = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        
        # 3. RNN (Deterministic Path)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # 4. Posterior Encoder (Inference Path)
        self.fc_mu = nn.Linear(self.rnn_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.rnn_hidden_dim, self.latent_dim)

        # 5. Decoder (Predictor)
        # 输入: z_t + h_t
        self.decoder_input_dim = self.latent_dim + self.rnn_hidden_dim
        self.state_decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_dim)
        )

        # 6. Reward Predictor (Independent Head)
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.decoder_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.rnn_hidden_dim).to(device)

    def forward_step(self, obs_input, hidden_state):
        """
        单步前向传播
        """
        # 1. Embedding
        x = F.relu(self.fc_input(obs_input))
        
        # 2. RNN Update: h_{t-1} -> h_t
        # 确保 hidden_state 维度正确 [B, H]
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        next_hidden = self.rnn(x, h_in)
        
        # 3. Predict Latent Distribution P(z_t | h_t)
        mu = self.fc_mu(next_hidden)
        logvar = self.fc_logvar(next_hidden)
        
        return next_hidden, (mu, logvar)

    def sample_latents(self, dist_params, num_samples=1):
        """
        重参数化采样 z
        返回: [Samples, Batch, Latent]
        """
        mu, logvar = dist_params
        std = torch.exp(0.5 * logvar)
        
        # [Samples, Batch, Latent]
        eps = torch.randn(num_samples, *mu.shape, device=mu.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        
        # 为了兼容旧代码 (如果调用方不处理 Samples 维度，我们自动 squeeze)
        # 但如果是因果推理 (需要多采样)，调用方应显式处理
        if num_samples == 1:
            z = z.squeeze(0)
            
        return z

    def reward_head(self, feature_input):
        """
        暴露给外部的 Reward 预测接口
        feature_input: cat([z, h], dim=-1)
        """
        return self.reward_predictor(feature_input)

    def decode(self, z, hidden_state):
        """
        手动解码预测 Next State (兼容旧接口)
        """
        inp = torch.cat([z, hidden_state], dim=-1)
        pred_next_state = self.state_decoder(inp)
        return pred_next_state

    def compute_loss(self, z_samples, hidden_state, target_state, dist_params, target_reward=None):
        """
        计算 WM 的所有 Loss
        增强: 支持 z_samples 为 [Samples, Batch, Latent] 的多采样计算
        """
        mu, logvar = dist_params
        
        # 维度适配
        # 如果 z 是 [B, L] (单次采样)，扩展为 [1, B, L]
        if z_samples.dim() == 2:
            z_samples = z_samples.unsqueeze(0)
        
        num_samples = z_samples.shape[0]
        
        # 扩展 hidden 和 targets 以匹配采样数 (Broadcasting)
        # hidden: [B, H] -> [1, B, H] -> [S, B, H]
        h_exp = hidden_state.unsqueeze(0).expand(num_samples, -1, -1)
        # target_state: [B, S] -> [S, B, S]
        s_target_exp = target_state.unsqueeze(0).expand(num_samples, -1, -1)
        
        # 1. Decode & Recon Loss
        decoder_input = torch.cat([z_samples, h_exp], dim=-1) # [S, B, L+H]
        pred_next_state = self.state_decoder(decoder_input)   # [S, B, State_Dim]
        
        # MSE over features, Mean over samples
        # reduction='none' -> [S, B, S_dim]
        recon_loss = F.mse_loss(pred_next_state, s_target_exp, reduction='none')
        recon_loss = recon_loss.sum(dim=-1).mean(dim=0) # [B]
        
        # 2. Reward Loss
        reward_loss = torch.zeros_like(recon_loss)
        if target_reward is not None:
            # target_reward: [B, 1] -> [S, B, 1]
            r_target_exp = target_reward.unsqueeze(0).expand(num_samples, -1, -1)
            
            pred_reward = self.reward_predictor(decoder_input) # [S, B, 1]
            
            r_loss = F.mse_loss(pred_reward, r_target_exp, reduction='none') # [S, B, 1]
            reward_loss = r_loss.sum(dim=-1).mean(dim=0) # [B]
        
        # 3. KL Loss (Analytical)
        # -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) # [B]
        
        return recon_loss, kl_loss, reward_loss

    def predict(self, obs_input, hidden_state, use_mean=True):
        """
        兼容旧代码的单步预测接口
        """
        next_hidden, (mu, logvar) = self.forward_step(obs_input, hidden_state)
        
        z = mu if use_mean else self.sample_latents((mu, logvar), num_samples=1)
        # sample_latents 可能会 squeeze，这里确保一下逻辑
        if z.dim() == 3: z = z.squeeze(0)
            
        inp = torch.cat([z, next_hidden], dim=-1)
        pred_next_state = self.state_decoder(inp)
        
        return pred_next_state, next_hidden