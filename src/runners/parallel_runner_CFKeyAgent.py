"""
并行环境运行器（ParallelRunner）

该模块提供了一个轻量的基于子进程的并行环境封装，灵感来自
OpenAI Baselines 中的 SubprocVecEnv。主要用途是在多个子进程中并行
运行环境实例，从而在多智能体训练时以批量方式收集经验。

主要组成：
- ParallelRunner: 在主进程中管理多个子进程和收集 batch 数据。
- env_worker: 子进程函数，接收主进程命令（reset、step、get_stats 等）并执行。
- CloudpickleWrapper: 用于对可调用对象进行序列化，避免 multiprocessing 使用 pickle 时出错。

注意：本文件只对代码添加中文注释，不更改原有逻辑。
"""

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
# ... (在 import numpy as np 之后)
import torch
from world_model_for_key_agent.world.vae_encoder import VAEEncoder 
from world_model_for_key_agent.world.vae import WorldModelVAE
from world_model_for_key_agent.world.res_vae import WorldModelResidualVAE
import torch.nn.functional as F

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class CFKeyAgentParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        # 创建 pipe 对：parent_conns 在主进程用于和每个子进程通信，worker_conns 给子进程使用
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])

        # 从环境注册表中获取环境工厂函数（不实例化），用于序列化后在子进程中创建环境
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            # CloudpickleWrapper 用于跨进程传递可调用对象（partial(env_fn, **env_args)）
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        # 将子进程设为守护进程并启动
        for p in self.ps:
            p.daemon = True
            p.start()

        # 向第一个子进程请求环境信息（假设每个环境的信息是一致的）
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        # --- VAE 初始化 (兼容两种模式) ---
        self.use_critical_agent = getattr(self.args, "use_critical_agent_obs", False)
        self.is_online_training = getattr(self.args, "use_online_vae_training", False)
        # 消融开关
        self.use_ablation = getattr(self.args, "use_ablation", False)

        self.vae_model = None # VAE 模型实例
        
        if self.use_critical_agent:
            self.logger.console_logger.info("Initializing VAE components...")
            # 1. 设置 VAE 运行所需的环境参数
            self._setup_vae_env_info() 
            
           # 2. (新) 获取 VAE 激活时间点 (用于时间门控)
            self.vae_activation_t = getattr(self.args, "vae_activatio", 0)
            self.logger.console_logger.info(f"VAE Critical Agent signal will ACTIVATE at t_env={self.vae_activation_t}")

            # === [消融]] 消融模式逻辑分支 ===
            if self.use_ablation:
                self.logger.console_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.logger.console_logger.info("!!! [ABLATION MODE] RANDOM CRITICAL AGENT !!!")
                self.logger.console_logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.logger.console_logger.info("VAE Model loading SKIPPED. Will use np.random instead.")
            # ============================

            # 3. 根据模式决定是“现在加载”还是“等待注入”
            elif self.is_online_training:
                # 模式2：在线训练。等待 Learner 注入模型
                self.logger.console_logger.info("VAE running in ONLINE mode (waiting for model from Learner).")
            else:
                # 模式1：离线推理。现在从磁盘加载预训练的 VAEEncoder
                self.logger.console_logger.info("VAE running in OFFLINE mode (loading pre-trained Encoder).")
                self._load_pretrained_vae_encoder()
        else:
            self.logger.console_logger.info("Not using VAE Encoder.")
            
        # 为所有并行环境存储上一步的观测
        self.last_obs_np = None 
        # --- VAE 初始化结束 ---

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

 
        # # 1. 存储 t=0 的观测
        # if self.use_critical_agent:
        #     # 将 obs 列表 (list of (n_agents, obs_shape)) 
        #     # 转换为 (batch_size, n_agents, obs_shape) 的 np 数组
        #     self.last_obs_np = np.stack(pre_transition_data["obs"], axis=0)
            
        #     # 2. t=0 时，关键智能体默认为 0
        #     crit_id_t_np = np.zeros((self.batch_size, 1), dtype=int)
        #     pre_transition_data["critical_id"] = crit_id_t_np

         # --- (已修改) 存储 t=0 的观测和标志 ---
        if self.use_critical_agent:
            self.last_obs_np = np.stack(pre_transition_data["obs"], axis=0)
            
            # t=0 时，关键智能体默认为 0
            crit_id_t_np = np.zeros((self.batch_size, 1), dtype=int)
            pre_transition_data["critical_id"] = crit_id_t_np
            
            # (新) t=0 时，标志位默认为 0 (未激活)
            crit_id_active_flag_np = np.zeros((self.batch_size, 1), dtype=int)
            pre_transition_data["critical_id_active"] = crit_id_active_flag_np
        # --- (修改结束) ---
    

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        #计算 VAE 预测误差的列表
        vae_test_mse_list = []
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # 1. MAC 决策 (t 时刻)
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            # 将 actions 从 GPU（或当前 device）移动到 CPU 并转成 numpy 数组以传入子进程
            # 注意：actions 需要确保不是 requires_grad=True，否则 .numpy() 会报错（应先 detach()）
            cpu_actions = actions.to("cpu").numpy()

            # 2. 存储 t 时刻的动作
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 3. (新) 为 VAE 准备 t 时刻的完整动作数组
            # (这必须在 t 时刻的 post_transition_data 之前完成)
            if self.use_critical_agent:
                # full_actions_t_np 存储 t 时刻的动作
                full_actions_t_np = np.zeros((self.batch_size, self.n_agents), dtype=int)
                action_idx_tracker = 0
                for idx in range(self.batch_size):
                    if idx in envs_not_terminated:
                        full_actions_t_np[idx] = cpu_actions[action_idx_tracker]
                        action_idx_tracker += 1
                # (已终止的环境动作保持为 0，这符合 VAE 的 "dead" 逻辑)
            
            # 4. 发送 t 时刻的动作到子进程
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env     
            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # 5. 准备 t 时刻的 Post-data 和 t+1 时刻的 Pre-data
            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

           # 6. 从子进程接收 t 时刻的 (r, d) 和 t+1 时刻的 (s, o, a)
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # t 时刻的 Post-data
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # t+1 时刻的 Pre-data
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # 7. 存储 t 时刻的 post-transition data
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # 8. 推进到 t+1
            self.t += 1

             # 9. (已修改) VAE 计算 (带时间门控)
            # 9. (新) VAE 计算
            if self.use_critical_agent:
                # 检查时间门控
                if self.t_env >= self.vae_activation_t:
                    # === [核心修改] 传入 envs_not_terminated 索引 ===
                    # 这样函数内部只会计算那些还活着的战局
                    crit_id_tplus1_np = self._batch_find_critical_agents(
                        self.last_obs_np, 
                        full_actions_t_np, 
                        envs_not_terminated # <--- 传入活跃环境索引列表
                    )
                    
                    crit_id_active_flag_np = np.ones((self.batch_size, 1), dtype=int)
                else:
                    crit_id_tplus1_np = np.zeros((self.batch_size, 1), dtype=int)
                    crit_id_active_flag_np = np.zeros((self.batch_size, 1), dtype=int)

                # (后续逻辑保持不变，继续过滤并填入 pre_transition_data)
                filtered_crit_id_np = crit_id_tplus1_np[envs_not_terminated]
                filtered_flag_np = crit_id_active_flag_np[envs_not_terminated]
                
                pre_transition_data["critical_id"] = filtered_crit_id_np
                pre_transition_data["critical_id_active"] = filtered_flag_np
            
           # === [计算预测准确度] 计算 VAE 预测误差 (仅在测试模式 + 在线训练开启时(离线模式没有加载decoder)) ===
            # 此时: 
            # self.last_obs_np 还是 t 时刻的 obs
            # full_actions_t_np 是 t 时刻的 action
            # pre_transition_data["state"] 是刚刚收到的 t+1 时刻的 state
            if test_mode and self.is_online_training and self.vae_model is not None:
                if len(envs_not_terminated) > 0:
                    # 1. 提取活跃环境的数据
                    active_obs = self.last_obs_np[envs_not_terminated]
                    active_actions = full_actions_t_np[envs_not_terminated]
                    # pre_transition_data["state"] 本身就是对应 envs_not_terminated 的列表
                    active_next_state = np.stack(pre_transition_data["state"], axis=0)
                    
                    # 2. 计算误差
                    mse_val = self._compute_vae_mse(active_obs, active_actions, active_next_state)
                    vae_test_mse_list.append(mse_val)
            # ==============================================================
            # 11. 存储 t+1 时刻的 pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        # === [遗漏的修复] 将 MSE 写入 stats ===
        if test_mode and len(vae_test_mse_list) > 0:
            mean_mse = np.mean(vae_test_mse_list)
            # 写入 cur_stats，Key 建议用 "vae_mse"
            # 注意：_log 会除以 n_episodes，所以这里乘上 batch_size 以便还原平均值
            cur_stats["vae_mse"] = cur_stats.get("vae_mse", 0) + mean_mse * self.batch_size
        # =====================================

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    # --- (新) VAE 辅助函数 ---
    def set_vae_model(self, vae_model_instance):
        """ (新) 允许 run.py 注入 Learner 的 VAE 模型 (在线模式) """
        if self.is_online_training:
            self.vae_model = vae_model_instance
            self.logger.console_logger.info("VAE Model (full) has been successfully linked from Learner.")
        else:
            self.logger.console_logger.warning("set_vae_model called, but Runner is not in ONLINE mode. Ignoring.")
        
    def _setup_vae_env_info(self):
        """ (新) 只加载 VAE 运行所需的环境参数 """
        try:
            self.n_agents = self.env_info["n_agents"]
            self.obs_shape = self.env_info["obs_shape"]
            self.n_actions = self.env_info["n_actions"]
            self.stop_action_id = self.args.stop_action_id
            
            # VAE 输入维度 (两种模式都需要)
            self.vae_input_dim = (self.n_agents * self.obs_shape) + (self.n_agents * self.n_actions)

        except KeyError as e:
            self.logger.console_logger.error(f"Error: 键 {e} 未在 env_info 或 args 中找到。")
            self.use_critical_agent = False
        except Exception as e:
            self.logger.console_logger.error(f"Error in _setup_vae_env_info: {e}")
            self.use_critical_agent = False

    def _load_pretrained_vae_encoder(self):
        """ (新) 加载预训练的 VAEEncoder (离线模式) """
        try:
            # 初始化 VAEEncoder (轻量级)
            self.vae_model = VAEEncoder(
                input_dim=self.vae_input_dim,
                latent_dim=self.args.vae_latent_dim,
                hidden_dim=self.args.vae_hidden_dim
            ).to(self.args.device)
            
            # 从磁盘加载
            self.vae_model.load_state_dict(torch.load(
                self.args.vae_encoder_path, map_location=self.args.device
            ))
            self.vae_model.eval() # 设为评估模式
            self.logger.console_logger.info(f"Loaded PRE-TRAINED VAE Encoder from {self.args.vae_encoder_path}")
        except FileNotFoundError:
            self.logger.console_logger.error(f"!!!!!!!! FAILED TO LOAD PRE-TRAINED VAE ENCODER !!!!!!!!")
            self.logger.console_logger.error(f"File not found at: {self.args.vae_encoder_path}")
            self.use_critical_agent = False
        except Exception as e:
            self.logger.console_logger.error(f"!!!!!!!! FAILED TO LOAD PRE-TRAINED VAE ENCODER !!!!!!!!")
            self.logger.console_logger.error(f"Error: {e}")
            self.use_critical_agent = False

    def _format_encoder_input(self, obs, actions):
        obs_flat = obs.flatten() 
        actions_squeezed = actions.squeeze().astype(int) 
        actions_one_hot = np.eye(self.n_actions)[actions_squeezed]
        actions_one_hot_flat = actions_one_hot.flatten() 
        encoder_input = np.concatenate([obs_flat, actions_one_hot_flat])
        return torch.tensor(encoder_input, dtype=torch.float32).unsqueeze(0).to(self.args.device)

    @torch.no_grad() 
    def _get_encoder_dist(self, obs, actions):
        # 安全检查
        if self.vae_model is None:
            # VAE 尚未被 Learner 注入 (在线模式) 或 加载失败 (离线模式) ,填0
            fake_mu = torch.zeros((1, self.args.vae_latent_dim), device=self.args.device)
            fake_log_var = torch.zeros((1, self.args.vae_latent_dim), device=self.args.device)
            return fake_mu, fake_log_var

        x_tensor = self._format_encoder_input(obs, actions)
        
        #  兼容两种模式
        if self.is_online_training:
            # 模式2 (在线): 我们持有 WorldModelVAE, 调用 .encode()
            mu, log_var = self.vae_model.encode(x_tensor) 
        else:
            # 模式1 (离线): 我们持有 VAEEncoder, 直接调用 (即 .forward())
            mu, log_var = self.vae_model(x_tensor)
        
        return mu, log_var

    @staticmethod
    def _kl_divergence(mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
        return torch.sum(kl, dim=1) 

    @torch.no_grad()
    def _find_critical_agent(self, obs_t_np, actions_t_np):
        kl_divergences = []
        mu_orig, log_var_orig = self._get_encoder_dist(obs_t_np, actions_t_np)
        for i in range(self.n_agents):
            original_agent_action = actions_t_np[i]
            if original_agent_action == 0:
                kl_divergences.append(float('-inf')) 
                continue
            actions_t_cf = np.copy(actions_t_np)
            actions_t_cf[i] = self.stop_action_id 
            mu_cf, log_var_cf = self._get_encoder_dist(obs_t_np, actions_t_cf)
            kl_div = self._kl_divergence(mu_orig, log_var_orig, mu_cf, log_var_cf)
            kl_divergences.append(kl_div.item())
        critical_agent_id = np.argmax(kl_divergences)
        if kl_divergences[critical_agent_id] == float('-inf'):
            return 0 
        return critical_agent_id
    
    @torch.no_grad()
    def _batch_find_critical_agents(self, obs_np, actions_np, active_indices):
        """
        [修复+加速版] 
        1. 仅对 active_indices (活跃环境) 进行推理，跳过已终止环境（大幅提速）。
        2. 兼容 VAEEncoder (离线) 和 WorldModelVAE (在线) 的接口差异。
        """
        # 如果没有活跃环境，或者模型没加载，直接返回全0
        if len(active_indices) == 0 or self.vae_model is None:
             return np.zeros((self.batch_size, 1), dtype=int)
        

        # === [修改] 消融实验：随机模式 ===
        if self.use_ablation:
            n_active = len(active_indices)
            # 在 [0, n_agents) 范围内均匀随机选择 ID
            random_ids = np.random.randint(0, self.n_agents, size=(n_active, 1))
            
            # 映射回完整的 batch (未活跃的保持为0)
            final_critical_ids = np.zeros((self.batch_size, 1), dtype=int)
            final_critical_ids[active_indices] = random_ids
            return final_critical_ids
        # ==============================

        # --- 1. 活跃切片 (Slicing) ---
        # 只提取活着的环境数据，大幅减少 GPU 计算量
        active_obs = obs_np[active_indices]         # (n_active, n_agents, obs_shape)
        active_actions = actions_np[active_indices] # (n_active, n_agents)
        
        n_active = len(active_indices)
        n_agents = self.n_agents
        n_actions = self.n_actions

        # --- 2. 数据准备 (CPU -> GPU) ---
        obs_flat = torch.tensor(active_obs.reshape(n_active, -1), dtype=torch.float32, device=self.args.device)
        actions_tensor = torch.tensor(active_actions, dtype=torch.long, device=self.args.device)
        actions_onehot = F.one_hot(actions_tensor, num_classes=n_actions).view(n_active, -1).float()
        
        original_input = torch.cat([obs_flat, actions_onehot], dim=1)

        # --- 3. 计算原始 Latent (兼容离线/在线接口) ---
        if self.is_online_training:
            mu_orig, log_var_orig = self.vae_model.encode(original_input) # 在线模式接口
        else:
            mu_orig, log_var_orig = self.vae_model(original_input)        # 离线模式接口
        
        # --- 4. 构造反事实输入 ---
        # 复制 active 样本
        obs_repeated = obs_flat.repeat_interleave(n_agents, dim=0) 
        actions_repeated = actions_tensor.repeat_interleave(n_agents, dim=0)
        
        # 构造 Mask 修改动作
        agent_indices = torch.arange(n_agents, device=self.args.device).repeat(n_active)
        # 将对应 agent 的动作修改为 stop
        actions_repeated[torch.arange(n_active * n_agents), agent_indices] = self.stop_action_id
        
        actions_cf_onehot = F.one_hot(actions_repeated, num_classes=n_actions).view(n_active * n_agents, -1).float()
        cf_inputs = torch.cat([obs_repeated, actions_cf_onehot], dim=1)
        
        # --- 5. 批量反事实推理 (兼容接口) ---
        if self.is_online_training:
            mu_cf, log_var_cf = self.vae_model.encode(cf_inputs)
        else:
            mu_cf, log_var_cf = self.vae_model(cf_inputs)
        
        # --- 6. 计算 KL 散度 ---
        mu_orig_expanded = mu_orig.repeat_interleave(n_agents, dim=0)
        log_var_orig_expanded = log_var_orig.repeat_interleave(n_agents, dim=0)
        
        kl_divs = self._kl_divergence(mu_orig_expanded, log_var_orig_expanded, mu_cf, log_var_cf)
        kl_matrix = kl_divs.view(n_active, n_agents)
        
        # Mask 掉原本就是 Stop 的动作
        # Mask 掉原本就是 Stop 的动作 (Action=stop_id) 
        # 以及 ！！！死掉的智能体 (Action=0)！！！
        is_stop_or_dead = (actions_tensor == self.stop_action_id) | (actions_tensor == 0)
        kl_matrix[is_stop_or_dead] = float('-inf')
        
        # 取最大值
        active_critical_ids = torch.argmax(kl_matrix, dim=1).unsqueeze(1).cpu().numpy()
        
        # --- 7. 还原完整 Batch ---
        final_critical_ids = np.zeros((self.batch_size, 1), dtype=int)
        final_critical_ids[active_indices] = active_critical_ids
        
        return final_critical_ids
    
    @torch.no_grad()
    def _compute_vae_mse(self, obs_np, actions_np, next_state_np):
        """
        计算 VAE 对未来状态预测的 MSE (均方误差)
        """
        n_samples = obs_np.shape[0]
        # 1. 数据转换
        # obs 已经是 (n_samples, n_agents * obs_shape) -> 2D
        obs = torch.tensor(obs_np.reshape(n_samples, -1), dtype=torch.float32, device=self.args.device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.args.device)
        next_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.args.device)
        
        # 2. 构造输入
        # actions_onehot 初始形状: (n_samples, n_agents, n_actions) -> 3D
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        
        # === [核心修复] 展平 actions_onehot 为 2D (n_samples, n_agents * n_actions) ===
        actions_onehot = actions_onehot.view(n_samples, -1) 
        # ========================================================================

        # 现在 obs 是 2D，actions_onehot 也是 2D，可以拼接了
        inputs = torch.cat([obs, actions_onehot], dim=1)
        
        # 3. VAE 前向预测
        recon_state, _, _ = self.vae_model(inputs)
        
        # 4. 计算 MSE
        mse = F.mse_loss(recon_state, next_state).item()
        return mse
    
    # --- (新) 辅助函数结束 ---


def env_worker(remote, env_fn):
    # 在子进程中创建环境并循环等待主进程命令
    # env_fn 是 CloudpickleWrapper(partial(env_factory, **env_args))，因此取出 .x() 来实例化环境
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            # data 中包含每个 agent 的动作，执行一步并返回本步的 reward/terminated/info
            actions = data
            # 注意：env.step 的返回格式需与这里的解包方式兼容（reward, terminated, env_info）
            reward, terminated, env_info = env.step(actions)
            # 获取下一时刻用于决策的观测数据
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # 下一时刻需要的观测数据
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # 本时刻步的信息
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            # 重置环境并返回初始观测
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            # 关闭环境并结束循环
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            # 返回环境静态信息（例如 episode_limit、n_agents 等）
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            # 返回统计信息（如胜负、伤害等额外信息）
            remote.send(env.get_stats())
        else:
            # 未实现的命令
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

