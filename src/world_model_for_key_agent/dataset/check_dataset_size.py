import os
import glob
import math
import random
import torch
from torch.utils.data import ConcatDataset

# 确保 vae_dataset.py 在同一个目录中
from vae_dataset import OfflineTrajectoryDataset 

# --- 您需要配置的参数 ---
DATA_ROOT_DIR = "/home/zhoujia/code/dataset/results/offline_data/3m"
TEST_SPLIT_RATIO = 0.1
BATCH_SIZE = 128
# -------------------------

def main():
    file_pattern = os.path.join(DATA_ROOT_DIR, "**", "*.h5")
    all_buffer_files = glob.glob(file_pattern, recursive=True)
    
    if not all_buffer_files:
        print(f"Error: No HDF5 files found matching pattern {file_pattern}")
        return
        
    print(f"Found {len(all_buffer_files)} total data files.")
    
    random.shuffle(all_buffer_files)
    split_idx = math.ceil(len(all_buffer_files) * (1 - TEST_SPLIT_RATIO))
    train_files = all_buffer_files[:split_idx]
    
    print(f"Splitting data: {len(train_files)} training files.")

    print("\nLoading training datasets (this may take a moment)...")
    train_datasets = []
    for f in train_files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            # 注意：OfflineTrajectoryDataset 会打印它自己的日志
            train_datasets.append(OfflineTrajectoryDataset(f)) 
            
    if not train_datasets:
        print("Error: No training data loaded successfully.")
        return
        
    train_dataset = ConcatDataset(train_datasets)
    
    total_transitions = len(train_dataset)
    batches_per_epoch = math.ceil(total_transitions / BATCH_SIZE)
    
    print("\n--- Dataset Size Report ---")
    print(f"Total Training Transitions: {total_transitions}")
    print(f"Batch Size (BATCH_SIZE):    {BATCH_SIZE}")
    print(f"Batches (BS) per Epoch:     {batches_per_epoch}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()