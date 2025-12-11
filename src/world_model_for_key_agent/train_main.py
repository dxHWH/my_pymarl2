import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import glob
import random
import math

# --- 从本地文件中导入所有构建块 ---
from world.vae import WorldModelVAE
from dataset.vae_dataset import OfflineTrajectoryDataset
from vae_trainer import VAETrainer
from vae_logger import setup_logging 

def main():
    # --- (新) 0. 配置日志 ---
    logger = setup_logging()

    # --- 1. 配置 (Config) ---
    
    # 数据配置
    DATA_ROOT_DIR = "/home/zhoujia/code/dataset/results/offline_data/3s5z_vs_3s6z/"
    TEST_SPLIT_RATIO = 0.1 
    
    # VAE 模型配置
    LATENT_DIM = 64
    HIDDEN_DIM = 512
    
    # --- 训练配置 ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    TOTAL_TRAIN_STEPS = 200000
    EVAL_EVERY_N_STEPS = 4000  
    EVAL_N_BATCHES = 1000
    
    # 保存路径配置
    BASE_MODEL_SAVE_DIR = "./vae_models" 
    MAP_FOLDER_NAME = os.path.basename(DATA_ROOT_DIR.strip(os.sep)) 
    FINAL_MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, MAP_FOLDER_NAME)
    MAP_NAME = "3s5z_vs_3s6z_all_data" 
    
    # -------------------------------------------------

    # --- 2. 设置 (Setup) ---

    os.makedirs(FINAL_MODEL_SAVE_DIR, exist_ok=True) 
    logger.info(f"Models will be saved to: {FINAL_MODEL_SAVE_DIR}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 3. 加载数据 (Data Loading) ---
    
    file_pattern = os.path.join(DATA_ROOT_DIR, "**", "*.h5")
    all_buffer_files = glob.glob(file_pattern, recursive=True)

    if not all_buffer_files:
        logger.error(f"Error: No HDF5 files found matching pattern {file_pattern}")
        return
        
    logger.info(f"Found {len(all_buffer_files)} total data files.")
    
    random.shuffle(all_buffer_files)
    # split_idx = math.ceil(len(all_buffer_files) * (1 - TEST_SPLIT_RATIO))

    n_files = len(all_buffer_files)
    
    if n_files < 2:
        # 如果 0 或 1 个文件, 无法创建测试集
        logger.warning(f"Found only {n_files} file. Using all for training. No test set will be created.")
        split_idx = n_files
    else:
        # 我们有 2 个或更多文件
        # 1. 优先计算测试集需要多少文件
        num_test_files = math.ceil(n_files * TEST_SPLIT_RATIO)
        
        # 2. 计算训练集的分割点
        split_idx = n_files - num_test_files
        
        # 3. 健全性检查:
        # 如果计算结果导致所有文件都分给了测试集 (例如 n_files=5, ratio=0.9)
        # 强制至少保留 1 个文件用于训练
        if split_idx == 0:
            split_idx = 1
    
    train_files = all_buffer_files[:split_idx]
    test_files = all_buffer_files[split_idx:]
    
    logger.info(f"Splitting data: {len(train_files)} training files, {len(test_files)} test files.")

    # 创建训练集
    train_datasets = [OfflineTrajectoryDataset(f) for f in train_files if os.path.exists(f)]
    if not train_datasets:
        logger.error("Error: No training data loaded successfully.")
        return
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # (优化)
    logger.info(f"Total training transitions: {len(train_dataset)}")
    
    # 创建测试集
    test_datasets = [OfflineTrajectoryDataset(f) for f in test_files if os.path.exists(f)]
    test_dataloader = None
    if test_datasets:
        test_dataset = ConcatDataset(test_datasets)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True) # (优化)
        logger.info(f"Total test transitions: {len(test_dataset)}")
    else:
        logger.warning("Warning: No test data loaded. Evaluation will be skipped.")

    # --- 4. 初始化对象 (Initialization) ---
    
    sample_dataset = train_datasets[0] 
    INPUT_DIM = (sample_dataset.n_agents * sample_dataset.obs_shape) + (sample_dataset.n_agents * sample_dataset.n_actions)
    STATE_SHAPE = sample_dataset.state_shape
    
    model = WorldModelVAE(
        input_dim=INPUT_DIM,
        state_dim=STATE_SHAPE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"VAE Model initialized.")
    logger.info(f"  Encoder Input Dim: {INPUT_DIM}")
    logger.info(f"  Decoder Output Dim: {STATE_SHAPE}")

    # --- 5. 运行 (Run) ---
    
    # --- (已更新) 配置训练器 ---
    trainer_config = {
        'total_train_steps': TOTAL_TRAIN_STEPS,
        'eval_every_n_steps': EVAL_EVERY_N_STEPS,
        'eval_n_batches': EVAL_N_BATCHES, 
        'save_dir': FINAL_MODEL_SAVE_DIR,
        'model_name': MAP_NAME
    }
    
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        config=trainer_config
    )
    
    # 启动训练！
    trainer.run_training()


if __name__ == "__main__":
    main()