from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch
from world_model_for_key_agent.world.vae_encoder import VAEEncoder


class CFKeyAgentEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # 是否使用智能体预测器 VAE 编码器
        self.use_critical_agent = getattr(self.args, "use_critical_agent_obs", False)
        if self.use_critical_agent:
            self.logger.console_logger.info("Initializing VAE Encoder for critical agent identification.")
            self._setup_vae()
        else:
            self.logger.console_logger.info("Not using VAE Encoder for critical agent identification.")
        # 记录上一步的观测
        self.last_obs_np = None
        self.last_actions_np = None
        self.last_critical_id = 0

    def _setup_vae(self):
        """ (新) 加载 VAE 编码器和所需的环境参数 """
        try:
            # 1. 从 args 获取环境/模型参数
            env_info = self.get_env_info()
            self.n_agents = env_info["n_agents"]
            self.obs_shape = env_info["obs_shape"]
            self.n_actions = env_info["n_actions"]

            self.stop_action_id = self.args.stop_action_id

            vae_input_dim = (self.n_agents * self.obs_shape) + (self.n_agents * self.n_actions)
            
            # 2. 初始化编码器模型
            self.vae_encoder = VAEEncoder(
                input_dim=vae_input_dim,
                latent_dim=self.args.vae_latent_dim,
                hidden_dim=self.args.vae_hidden_dim
            ).to(self.args.device)
            
            self.vae_encoder.load_state_dict(torch.load(
                self.args.vae_encoder_path, map_location=self.args.device
            ))
            self.vae_encoder.eval()
            self.logger.console_logger.info(f"Loaded VAE Encoder from {self.args.vae_encoder_path}")

        except KeyError as e:
            self.logger.console_logger.error(f"!!!!!!!! FAILED TO LOAD VAE ENCODER !!!!!!!!")
            self.logger.console_logger.error(f"Error: 键 {e} 未在 env_info 或 args 中找到。")
            self.logger.console_logger.error("请确保您的 default.yaml 和 env_info 包含所有必需的键。")
            self.use_critical_agent = False
        except Exception as e:
            self.logger.console_logger.error(f"!!!!!!!! FAILED TO LOAD VAE ENCODER !!!!!!!!")
            self.logger.console_logger.error(f"Error: {e}")
            self.logger.console_logger.error("请确保 default.yaml 中已正确配置 VAE 参数")
            self.logger.console_logger.error("(vae_encoder_path, vae_hidden_dim, vae_latent_dim, stop_action_id)")
            self.use_critical_agent = False # 加载失败则禁用


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

        # 重置VAE状态
        if self.use_critical_agent:
            self.last_critical_id = 0 # 默认 0
            #这里可能有点问题
            self.last_obs_np = np.array(self.env.get_obs(), dtype=np.float32) 
            self.last_actions_np = np.zeros((self.n_agents,), dtype=int) # (t=-1) 的虚拟动作

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            # 获取t时刻的观测
            # 1. 获取 obs 列表
            obs_t_list = self.env.get_obs()
            # 2. 将其转换为一个堆叠的 numpy 数组
            obs_t_np = np.array(obs_t_list, dtype=np.float32)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [obs_t_np] # <-- (已修正) 使用转换后的 np.array
            }

            # 将上一步 (t-1) 计算出的 critical_id 存入 batch
            if self.use_critical_agent:
                # self.last_critical_id 是在 t-1 循环的末尾计算的
                pre_transition_data["critical_id"] = [(self.last_critical_id,)] 

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, env_info = self.env.step(cpu_actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            # 计算VAE,算出t时刻的关键智能体，并赋值给 last_critical_id
            if self.use_critical_agent:
                actions_t_np = cpu_actions[0] # (n_agents,)
                
                # 使用 (obs_t, actions_t) 来计算 *下一个* critical_id
                #  "t时刻的关键智能体，t+1时刻去关注" 的假设
                self.last_critical_id = self._find_critical_agent(obs_t_np, actions_t_np)
                
            self.t += 1
        #最后一步的动作执行之后的反馈
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [np.array(self.env.get_obs(), dtype=np.float32)] # <-- (已修正) 同样在此处转换
        }
        if self.use_critical_agent:
            #对齐样本，理论上，这一步的critical_id不会被使用到
            last_data["critical_id"] = [(self.last_critical_id,)]
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
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
    def _format_encoder_input(self, obs, actions):
        """
        将 (obs, actions) numpy 数组格式化为模型所需的扁平张量。
        :param obs: (n_agents, obs_shape)
        :param actions: (n_agents,) 
        :return: (1, input_dim) 的 torch 张量
        """

        obs_flat = obs.flatten() 
        actions_squeezed = actions.squeeze().astype(int) 
        actions_one_hot = np.eye(self.n_actions)[actions_squeezed]
        actions_one_hot_flat = actions_one_hot.flatten() 
        
        encoder_input = np.concatenate([obs_flat, actions_one_hot_flat])
        
        return torch.tensor(encoder_input, dtype=torch.float32).unsqueeze(0).to(self.args.device)

    @torch.no_grad() 
    def _get_encoder_dist(self, obs, actions):
        x_tensor = self._format_encoder_input(obs, actions)
        mu, log_var = self.vae_encoder(x_tensor) 
        return mu, log_var

    @staticmethod
    def _kl_divergence(mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mu1 - mu2)**2) / var2 - 1.0)
        return torch.sum(kl, dim=1) 

    @torch.no_grad()
    def _find_critical_agent(self, obs_t_np, actions_t_np):
        """
        执行反事实推理，找到关键智能体。
        obs_t_np: (n_agents, obs_shape)
        actions_t_np: (n_agents,)
        """
        kl_divergences = []
        
        # 1. 计算原始分布
        mu_orig, log_var_orig = self._get_encoder_dist(obs_t_np, actions_t_np)

        # 2. 循环 n 个智能体
        for i in range(self.n_agents):
            original_agent_action = actions_t_np[i]
            
            # 3. 检查是否阵亡 (action 0)
            if original_agent_action == 0:
                kl_divergences.append(float('-inf')) 
                continue
            
            # 4. 创建反事实动作
            actions_t_cf = np.copy(actions_t_np)
            actions_t_cf[i] = self.stop_action_id 
            
            # 5. 计算反事实分布
            mu_cf, log_var_cf = self._get_encoder_dist(obs_t_np, actions_t_cf)
            
            # 6. 计算 KL
            kl_div = self._kl_divergence(mu_orig, log_var_orig, mu_cf, log_var_cf)
            kl_divergences.append(kl_div.item())

        # 7. 找到 ID
        critical_agent_id = np.argmax(kl_divergences)
        
        if kl_divergences[critical_agent_id] == float('-inf'):
            return 0 # 如果都阵亡了，默认关注 0 号
        
        return critical_agent_id
    # --- (新) 辅助函数结束 ---