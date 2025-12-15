from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        #运行的时间步
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

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

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0#设置轨迹总奖励为0
        #初始化controller隐藏层
        self.mac.init_hidden(batch_size=self.batch_size)

        #开始和环境交互，一条轨迹
        while not terminated:
            #记录环境当前状态、可用动作、每个智能体观测 
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # 更新信息到到当前batch中
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents 翻译：将整个batch的经验传递给智能体

            # 这里的batch是经验池
            # 这里的t是时间步
            # Receive the actions for each agent at this timestep in a batch of size 1：在一个batch大小为1的情况下，接收每个智能体在当前时间步的动作
            # 调用controller,根据环境、轨迹信息，选择动作
            # 这里的mac是controller
            actions, qvals = self.mac.select_actions_with_qvals(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            cpu_qvals = qvals.to("cpu").numpy()
            
            #环境步进
            reward, terminated, env_info = self.env.step(actions[0])
            #积累轨迹奖励
            episode_return += reward

            #记录环境当前状态、可用动作、每个智能体观测、动作、Q值、奖励、是否终止
            post_transition_data = {
                "actions": cpu_actions,
                "qvals": cpu_qvals,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            #更新到当前batch中
            self.batch.update(post_transition_data, ts=self.t)
            #时间步+1
            self.t += 1
        #本条轨迹的结束信息：最后状态的下一步状态、可用动作、每个智能体观测；用于后续计算时序差分损失
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, qvals = self.mac.select_actions_with_qvals(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        cpu_qvals = qvals.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions, "qvals": cpu_qvals}, ts=self.t)
        
        #记录当前轨迹的信息
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


        #返回这一批轨迹的具体数据，例如返回给runner.run()的调用
        #轨迹中的一步包含的信息：
        # {
        #     "state",
        #     "avail_actions",
        #     "obs",
        #     "actions",
        #     "qvals",  # 新增：每个智能体对所有动作的Q值
        #     "reward",
        #     "terminated",
        # }
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
