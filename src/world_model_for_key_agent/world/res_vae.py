import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 残差块定义 (核心增强组件) ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        """
        全连接残差块: x = x + f(x)
        包含 LayerNorm 以稳定深层训练。
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True), # 使用 LeakyReLU 保持负区间梯度
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # 残差相加后的激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        return self.activation(out)

# --- 2. 增强版 VAE 模型定义 ---
class WorldModelResidualVAE(nn.Module):
    def __init__(self, input_dim, state_dim, latent_dim, hidden_dim=1024, num_res_blocks=3):
        """
        增强版 VAE 世界模型。
        :param hidden_dim: 建议设为 512 或 1024
        :param num_res_blocks: 残差块数量，建议 2~4
        """
        super(WorldModelResidualVAE, self).__init__()

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

        # === 编码器 (Encoder) ===
        # 1. 特征投影层 (Input -> Hidden)
        self.encoder_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. 深层特征提取 (堆叠残差块)
        self.encoder_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        
        # 3. 潜在变量头 (Hidden -> Latent)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # === 解码器 (Decoder) ===
        # 1. 潜在投影层 (Latent -> Hidden)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. 深层重构 (堆叠残差块)
        self.decoder_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        
        # 3. 状态预测头 (Hidden -> State)
        self.decoder_output = nn.Linear(hidden_dim, state_dim)

    def encode(self, x):
        """ 编码器前向传播 """
        h = self.encoder_input(x)
        h = self.encoder_blocks(h)  # 通过残差网络
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
        h = self.decoder_input(z)
        h = self.decoder_blocks(h)
        return self.decoder_output(h)

    def forward(self, x):
        """ VAE 完整前向传播 (Training用) """
        # 1. 编码
        mu, log_var = self.encode(x)
        # 2. 采样
        z = self.reparameterize(mu, log_var)
        # 3. 解码
        recon_state = self.decode(z)
        return recon_state, mu, log_var

# --- 3. VAE 损失函数 ---
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """
    计算 VAE 损失 (ELBO)
    :param beta: KL 散度的权重因子 (用于 Beta-VAE 或 KL Annealing)
    """
    # 1. 重构损失 (MSE)
    # 使用 sum 后除以 batch_size (即 mean over batch, sum over features) 保持数值稳定性
    batch_size = x.size(0)
    RCE_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size

    # 2. KL 散度损失 (Analytical KLD)
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    KLD_loss = torch.sum(KLD_element) / batch_size

    # 3. 总损失
    total_loss = RCE_loss + beta * KLD_loss
    
    return total_loss, RCE_loss, KLD_loss