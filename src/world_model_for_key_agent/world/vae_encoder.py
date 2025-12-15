import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    """
    只包含 VAE 编码器部分的独立模型，用于推理。
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        """
        参数必须与您训练时的 WorldModelVAE 完全一致。
        :param input_dim: 编码器的输入维度 (obs_flat + actions_onehot_flat)
        :param latent_dim: 潜在向量 z 的维度
        :param hidden_dim: 隐藏层维度
        """
        super(VAEEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # --- 编码器 (Encoder) ---
        # 结构必须与 vae_model.py 中的完全一致
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """ 
        编码器前向传播 
        输入 x: 展平的 (o_t, a_t) 张量
        输出: (mu, log_var)
        """
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var