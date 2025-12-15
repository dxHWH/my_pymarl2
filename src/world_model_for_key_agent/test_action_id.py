import h5py
import numpy as np
import glob
import os

# --- !! 修改这里的路径 !! ---
# 指向您数据文件的根目录，脚本会检查第一个找到的 .h5 文件
DATA_ROOT_DIR = "/home/zhoujia/code/dataset/results/offline_data/3s5z_vs_3s6z/qmix__2025-05-07_23-59-17_full" 
# -------------------------

def verify_action_ids():
    print(f"正在搜索 {DATA_ROOT_DIR} 中的 .h5 文件...")
    file_pattern = os.path.join(DATA_ROOT_DIR, "**", "*.h5")
    all_files = glob.glob(file_pattern, recursive=True)
    
    if not all_files:
        print(f"错误：在 {DATA_ROOT_DIR} 中未找到任何 .h5 文件。")
        return

    file_to_check = all_files[0]
    print(f"--- 正在分析文件: {file_to_check} ---")
    
    try:
        f = h5py.File(file_to_check, 'r')
        
        # 加载 'avail_actions' 和 'filled'
        avail_actions = f['avail_actions'][:] # (N, T, n_agents, n_actions)
        filled = f['filled'][:]               # (N, T, 1)
        f.close()
        
        n_actions = avail_actions.shape[-1]
        print(f"检测到总动作数 (n_actions): {n_actions}")
        
        # --- 关键验证逻辑 ---
        
        # 1. 创建一个布尔掩码，只选择“有效”的时间步（即非填充数据）
        # filled.squeeze() 会得到 (N, T)
        # filled.squeeze() == 1 会得到一个 (N, T) 的布尔掩码
        # 我们用它来索引 avail_actions 的前两个维度
        
        # 首先，我们需要确保 filled 和 avail_actions 的 N, T 维度匹配
        # (N, T, n_agents, n_actions) -> (N*T, n_agents, n_actions)
        avail_actions_flat = avail_actions.reshape(-1, avail_actions.shape[2], avail_actions.shape[3])
        
        # (N, T, 1) -> (N*T, 1)
        filled_flat = filled.reshape(-1, 1)
        
        # 仅选择 filled == 1 的行
        valid_avail_actions = avail_actions_flat[filled_flat.squeeze() == 1]
        # valid_avail_actions 的 shape 是 (total_valid_steps, n_agents, n_actions)
        
        if valid_avail_actions.shape[0] == 0:
            print("错误：未在此文件中找到任何有效时间步。")
            return
            
        print(f"从 {valid_avail_actions.shape[0]} 个有效时间步中进行分析...")

        # 2. 检查每个动作 ID 的“平均可用率”
        print("\n--- 动作 ID 可用率 (1.0 = 100% 可用) ---")
        for i in range(n_actions):
            # (total_valid_steps, n_agents, n_actions) -> (total_valid_steps, n_agents)
            action_availability = valid_avail_actions[:, :, i]
            
            # 计算这个动作在所有智能体、所有有效时间步中的平均可用率
            mean_availability = np.mean(action_availability)
            
            print(f"Action ID {i}: {mean_availability:.4f}")

        # --- 结论 ---
        print("\n--- 结论 ---")
        if (np.mean(valid_avail_actions[:, :, 0]) > 0.99 and 
            np.mean(valid_avail_actions[:, :, 1]) > 0.99):
            print("验证成功！ID 0 和 ID 1 的可用率均为 ~100%。")
            print("请在您的 CONFIG 中使用:")
            print("  'stop_action_id': 1")
        else:
            print("验证失败。ID 0 和/或 ID 1 并非 100% 可用。")
            print("请仔细检查上面的可用率列表，找出哪个 ID 才是 'stop'。")
            
    except Exception as e:
        print(f"分析时出错: {e}")

if __name__ == "__main__":
    verify_action_ids()