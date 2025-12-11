# %%
import h5py
f = h5py.File('/home/zhoujia/code/dataset/results/offline_data/3m/qmix__2025-05-06_18-13-34_win_rate_0.25/part_0.h5', 'r')
list(f.keys())
# %%
traj_len = f['actions'].shape[0]
print('轨迹数量:', traj_len)
state = f['state'][:]
print('状态 shape:', state.shape)
actions = f['actions'][:]
print('动作 shape:', actions.shape)
rewards = f['reward'][:]
print('奖励 shape:', rewards.shape)
filled = f['filled'][:]
print('填充标志 shape:', filled.shape)
obs = f['obs'][:]
print('观测 shape:', obs.shape)
avail_actions = f['avail_actions'][:]
print('可用动作 shape:', avail_actions.shape)
probs = f['probs'][:]
print('动作概率 shape:', probs.shape) # 轨迹数，轨迹内时间步，智能体，动作维度
# %%

