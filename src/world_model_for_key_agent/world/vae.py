import torch
import torch.nn as nn
import torch.nn.functional as F

# --- VAE 模型定义 ---

class WorldModelVAE(nn.Module):
    def __init__(self, input_dim, state_dim, latent_dim, hidden_dim=128):
        """
        VAE 世界模型.
        :param input_dim: 编码器的输入维度 (obs_flat + actions_onehot_flat)
        :param state_dim: 解码器的输出维度 (state_shape)
        :param latent_dim: 潜在向量 z 的维度
        :param hidden_dim: 隐藏层维度
        """
        super(WorldModelVAE, self).__init__()

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

        # --- 编码器 (Encoder) ---
        # 输入: (obs_t, act_t) -> 输出: mu, log_var
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # --- 解码器 (Decoder) ---
        # 输入: z -> 输出: state_t+1_pred
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) # 输出重构的 state
        )

    def encode(self, x):
        """ 编码器前向传播 """
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """ 重参数化技巧 """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """ 解码器前向传播 """
        return self.decoder_net(z)

    def forward(self, x):
        """ VAE 完整前向传播 """
        # 1. 编码
        mu, log_var = self.encode(x)
        # 2. 采样
        z = self.reparameterize(mu, log_var)
        # 3. 解码
        recon_state = self.decode(z)
        return recon_state, mu, log_var

# --- VAE 损失函数 (ELBO) ---

def vae_loss_function(recon_x, x, mu, log_var, beta=0.075):
    """
    计算 VAE 损失 (ELBO)
    Loss = 重构损失 + Beta * KL散度损失
    :param recon_x: 重构的 state_t+1
    :param x: 真实的 state_t+1
    :param mu: 潜在均值
    :param log_var: 潜在对数方差
    :param beta: KL 散度的权重因子 (用于 Beta-VAE, 默认为 1)
    """
    # 1. 重构损失 (Reconstruction Loss)
    RCE_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL 散度损失 (Kullback-Leibler Divergence)
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # 3. 计算总损失
    total_loss = RCE_loss + beta * KLD_loss
    
    # 4.返回所有三个分量
    return total_loss, RCE_loss, KLD_loss 