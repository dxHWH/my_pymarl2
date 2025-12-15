import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

# --- 离线数据加载器 ---

class OfflineTrajectoryDataset(Dataset):
    """
    用于PyMARL2 HDF5 (.h5) 轨迹数据的 PyTorch Dataset.
    """
    def __init__(self, buffer_file):
        print(f"Loading data from HDF5 file: {buffer_file}...")
        
        self.transitions = []
        
        try:
            with h5py.File(buffer_file, 'r') as f:
                # --- 从 HDF5 文件加载所有需要的数据到内存 ---
                all_obs = f['obs'][:]
                all_state = f['state'][:]
                all_filled = f['filled'][:]
                
                if 'actions_onehot' in f:
                    all_actions_onehot = f['actions_onehot'][:]
                else:
                    print("Warning: 'actions_onehot' not found. Creating from 'actions'.")
                    all_actions = f['actions'][:] 
                    n_actions = f['avail_actions'].shape[-1] 
                    actions_squeezed = all_actions.squeeze(-1).astype(int)
                    all_actions_onehot = np.eye(n_actions)[actions_squeezed]

                # --- 自动推断环境参数 ---
                self.n_episodes = all_obs.shape[0]
                self.max_seq_len = all_obs.shape[1]
                self.n_agents = all_obs.shape[2]
                self.obs_shape = all_obs.shape[3]
                self.state_shape = all_state.shape[2]
                self.n_actions = all_actions_onehot.shape[3]
                
                print("--- Dataset Parameters Inferred ---")
                print(f"  N Episodes:    {self.n_episodes}")
                print(f"  Max Seq Len:   {self.max_seq_len}")
                print(f"  N Agents:      {self.n_agents}")
                print(f"  Obs Shape:     {self.obs_shape}")
                print(f"  State Shape:   {self.state_shape}")
                print(f"  N Actions:     {self.n_actions}")
                print("-----------------------------------")


                # --- 遍历所有数据，构建 (输入, 输出) 对 ---
                for i in range(self.n_episodes):
                    episode_length = int(all_filled[i].sum())

                    for t in range(episode_length - 1):
                        # 获得所有智能体的观测
                        obs_t = all_obs[i, t]
                        # 获得所有智能体的动作
                        actions_t_one_hot = all_actions_onehot[i, t]
                        # 获得下一个时间步的全局状态
                        state_t_plus_1 = all_state[i, t + 1]
                        
                        # 展平并拼接观测向量
                        obs_t_flat = obs_t.flatten()
                        # 展平并拼接动作向量
                        actions_t_one_hot_flat = actions_t_one_hot.flatten()
                        # 拼接成编码器输入
                        encoder_input = np.concatenate([obs_t_flat, actions_t_one_hot_flat])
                        self.transitions.append((encoder_input, state_t_plus_1))

        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            raise

        print(f"Successfully processed {len(self.transitions)} valid (o_t, a_t) -> s_t+1 transitions.")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        encoder_input, state_t_plus_1 = self.transitions[idx]
        
        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(state_t_plus_1, dtype=torch.float32)
        )