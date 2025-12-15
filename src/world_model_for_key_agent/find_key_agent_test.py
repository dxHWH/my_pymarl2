import torch
import torch.nn.functional as F # <-- (新) 导入 F (用于 L1 Loss)
import h5py
import numpy as np
import os
import logging
# --- (已更新) ---
# 我们现在导入完整的 VAE 模型，而不是 VAEEncoder
# 假设您的 vae_model.py 位于 'world' 文件夹下
from world.vae import WorldModelVAE 
# --------------------

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- !! 关键配置：请务必修改为您自己的参数 !! ---
CONFIG = {
    # --- 1. 路径配置 (已更新) ---
    # !! 必须指向您完整的 VAE 模型，而不是 encoder !!
    "world_model_path": "./vae_models/3s5z_vs_3s6z/vae_world_model_3s5z_vs_3s6z_all_data_best.pt", 
    "h5_data_path": "/home/zhoujia/code/dataset/results/offline_data/3s5z_vs_3s6z/qmix__2025-05-07_23-59-17_full/part_0.h5",

    # --- 2. VAE 模型架构参数 (必须与训练时完全一致) ---
    "hidden_dim": 512,  # 您在训练时使用的 HIDDEN_DIM (例如 512)
    "latent_dim": 64,   # 您在训练时使用的 LATENT_DIM (例如 64)

    # --- 3. 环境参数 (必须与训练时完全一致) ---
    "n_agents": 8,
    "obs_shape": 136,
    "n_actions": 15,
    "state_shape": 230,
    
    # --- 4. 关键智能体识别参数 ---
    "stop_action_id": 5, # !! 您确认过的 'stop' 动作 ID
}
# ---------------------------------------------------


class KeyAgentFinder:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # --- 1. 计算维度 ---
        self.n_agents = config['n_agents']
        self.obs_shape = config['obs_shape']
        self.n_actions = config['n_actions']
        self.state_shape = config['state_shape']
        self.stop_action_id = config['stop_action_id']
        
        self.input_dim = (self.n_agents * self.obs_shape) + (self.n_agents * self.n_actions)
        
        logger.info(f"Input Dim: {self.input_dim}, N-Agents: {self.n_agents}, N-Actions: {self.n_actions}")
        
        # --- 2. 加载模型 (已更新) ---
        # 加载完整的 WorldModelVAE
        self.model = WorldModelVAE(
            input_dim=self.input_dim,
            state_dim=self.state_shape, # <-- 需要 state_shape
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        try:
            # 加载完整的 state_dict
            self.model.load_state_dict(torch.load(config['world_model_path'], map_location=self.device))
            self.model.eval() 
            logger.info(f"Successfully loaded full VAE model from: {config['world_model_path']}")
        except FileNotFoundError:
            logger.error(f"Error: Model file not found at {config['world_model_path']}")
            raise
        except Exception as e:
            logger.error(f"Error loading model state_dict: {e}")
            logger.error("确保 WorldModelVAE 的架构与您保存的权重完全匹配。")
            raise

        # --- 3. 加载 H5 离线数据到内存 (已更新) ---
        logger.info(f"Loading H5 data from: {config['h5_data_path']}...")
        try:
            data = h5py.File(config['h5_data_path'], 'r')
            self.all_obs = data['obs'][:]        
            self.all_actions = data['actions'][:]  
            self.all_filled = data['filled'][:]
            self.all_state = data['state'][:] # <-- (新) 加载 state 真值
            data.close()
            logger.info(f"H5 data loaded into memory. Found {self.all_obs.shape[0]} episodes.")
        except FileNotFoundError:
            logger.error(f"Error: H5 data file not found at {config['h5_data_path']}")
            raise

    def _format_encoder_input(self, obs, actions):
        """ (此函数无需修改) """
        obs_flat = obs.flatten() 
        actions_squeezed = actions.squeeze().astype(int) 
        actions_one_hot = np.eye(self.n_actions)[actions_squeezed]
        actions_one_hot_flat = actions_one_hot.flatten() 
        encoder_input = np.concatenate([obs_flat, actions_one_hot_flat])
        return torch.tensor(encoder_input, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad() 
    def _get_encoder_dist(self, obs, actions):
        """ (已更新) 使用 self.model.encode """
        x_tensor = self._format_encoder_input(obs, actions)
        # (已更新) 调用 .encode() 而不是 .forward()
        mu, log_var = self.model.encode(x_tensor) 
        return mu, log_var

    # --- (新) 预测函数 ---
    @torch.no_grad()
    def _get_state_prediction(self, mu, log_var):
        """
        从潜在分布中获取一个确定的状态预测值。
        我们使用 mu (均值) 作为最可能的 z，以进行确定性预测。
        """
        # z = self.model.reparameterize(mu, log_var) # 随机采样
        z = mu # 确定性预测
        s_t_plus_1_pred = self.model.decode(z)
        return s_t_plus_1_pred

    @staticmethod
    def _kl_divergence(mu1, log_var1, mu2, log_var2):
        """ (此函数无需修改) """
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
        return torch.sum(kl, dim=1) 

    def find_critical_agent_at_timestamp(self, episode_idx, t):
        """
        (已更新) 执行反事实推理，并验证预测准确性。
        """
        logger.info(f"--- Analyzing Episode {episode_idx}, Timestamp {t} ---")
        
        # --- 1. 获取 t 时刻的真实数据 ---
        obs_t = self.all_obs[episode_idx, t]
        actions_t_orig = self.all_actions[episode_idx, t]
        # (新) 获取 t+1 时刻的真实状态
        state_t_plus_1_true_np = self.all_state[episode_idx, t + 1]
        state_t_plus_1_true = torch.tensor(state_t_plus_1_true_np, dtype=torch.float32).to(self.device)

        # --- 2. 计算“原始”分布 (所有智能体正常行动) ---
        mu_orig, log_var_orig = self._get_encoder_dist(obs_t, actions_t_orig)
        
        logger.info(f"Original Actions: {actions_t_orig.squeeze().tolist()}")
        logger.info(f"Original Dist (mu mean): {mu_orig.mean().item():.4f}")

        # --- 3. (新) 验证预测准确性 ---
        s_t_plus_1_pred = self._get_state_prediction(mu_orig, log_var_orig)
        
        # 计算 MAE 和 R²
        # 确保它们是 (N,) 或 (N, C) 形状，而不是 (1, N)
        pred_flat = s_t_plus_1_pred.squeeze()
        true_flat = state_t_plus_1_true.squeeze()

        mae = F.l1_loss(pred_flat, true_flat).item()
        ss_res = ((true_flat - pred_flat) ** 2).sum()
        ss_tot = ((true_flat - true_flat.mean()) ** 2).sum()
        r2 = (1.0 - (ss_res / (ss_tot + 1e-8))).item()
        
        logger.info(f"--- Prediction Accuracy (t -> t+1) ---")
        logger.info(f"Prediction MAE: {mae:.6f}  (R²: {r2:.6f})")
        logger.info(f"----------------------------------------")

        kl_divergences = []

        # --- 4. 循环 n 个智能体，执行反事实推理 ---
        for i in range(self.n_agents):
            
            original_agent_action = actions_t_orig.squeeze()[i]
            
            # 检查智能体是否阵亡 (action 0)
            if original_agent_action == 0:
                logger.info(f"  Agent {i} is dead (Action 0). Skipping KL calculation.")
                kl_divergences.append(float('-inf')) 
                continue
            
            actions_t_cf = np.copy(actions_t_orig)
            actions_t_cf[i] = self.stop_action_id 
            
            mu_cf, log_var_cf = self._get_encoder_dist(obs_t, actions_t_cf)
            
            kl_div = self._kl_divergence(mu_orig, log_var_orig, mu_cf, log_var_cf)
            
            kl_divergences.append(kl_div.item())
            logger.info(f"  Agent {i} (alive) stops -> KL Divergence: {kl_div.item():.6f}")

        # --- 5. 找到 KL 散度最大的智能体 ---
        critical_agent_id = np.argmax(kl_divergences)
        max_kl = kl_divergences[critical_agent_id]
        
        logger.info(f"--- Result ---")
        logger.info(f"KL Divergences: {kl_divergences}")
        
        if max_kl == float('-inf'):
            logger.info(f"All agents are dead. No critical agent found.")
            return -1, 0.0 
        else:
            logger.info(f"Critical Agent ID: {critical_agent_id} (with KL = {max_kl:.6f})")
            return critical_agent_id, max_kl

def main():
    # --- 1. 初始化 Finder ---
    try:
        finder = KeyAgentFinder(CONFIG)
    except Exception as e:
        logger.error(f"Failed to initialize finder. {e}")
        return

    # --- 2. 选择一个轨迹进行测试 ---
    EPISODE_TO_TEST = 123
    TIMESTAMP_TO_TEST = 9
    
    try:
        episode_length = int(finder.all_filled[EPISODE_TO_TEST].sum())
        
        # (已修复) 确保 t+1 是有效的
        if TIMESTAMP_TO_TEST >= episode_length - 1:
            logger.warning(f"Timestamp {TIMESTAMP_TO_TEST} is out of bounds for episode {EPISODE_TO_TEST} (length {episode_length}).")
            logger.warning(f"Defaulting to timestamp 5.")
            TIMESTAMP_TO_TEST = 5

        # --- 3. 运行验证 ---
        finder.find_critical_agent_at_timestamp(EPISODE_TO_TEST, TIMESTAMP_TO_TEST)
            
    except IndexError:
        logger.error(f"Episode {EPISODE_TO_TEST} is out of bounds for the loaded H5 file.")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()