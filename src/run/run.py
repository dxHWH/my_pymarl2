import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

#imoport runer和learner中穿件的字典
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        # tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs",args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

# 评测函数，进行一定轮次的测试
def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:#测试结果以SMAC录像的格式保存，linux不能打开，需要到打开windows客户端
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    # 根据配置文件创建runner，runer负责运行
    # 需要自己实现新的runer（leaner同理）需要先到runer下创建类，再在init中注册对应的字典内容，
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]#智能体数量
    args.n_actions = env_info["n_actions"]#动作数量
    args.state_shape = env_info["state_shape"]#状态维度
    args.obs_shape = env_info["obs_shape"]#观察维度
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    # scheme字典，记录 观测、状态、动作、奖励、终止等信息的维度，以确定神经网络的输入输出 
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        # "qvals": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        # "critical_id": {"vshape": (1,), "dtype": th.long},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        # "test": {"vshape": (1,),"episode_const":True, "dtype": th.uint8},
    }

     # (已修改) 添加 VAE 所需的 scheme
    if getattr(args, "use_critical_agent_obs", False):
        scheme["critical_id"] = {"vshape": (1,), "dtype": th.long}
        # (新) 添加一个标志位, 告诉 MAC 辅助信息是否“激活”
        scheme["critical_id_active"] = {"vshape": (1,), "dtype": th.long} 

    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # 经验回放池
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    # mac-MultiAgentController，mac负责控制智能体，创建流程与runer相同,
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme 初始化runer 初始化智能体的控制器
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 创建Learner，网络训练，创建流程与runer相同
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # 若使用cuda,将网络权重放到GPU上
    if args.use_cuda:
        learner.cuda()

     # --- (新) "单一模型引用" 链接逻辑 ---
    # 检查是否使用 VAE Learner 和 Runner
    if getattr(args, "use_online_vae_training", False) and getattr(args, "use_critical_agent_obs", False):
        if hasattr(learner, "vae_model") and hasattr(runner, "set_vae_model"):
            logger.console_logger.info("Linking Learner's (full) VAE model to Runner...")
            # 将 Learner 的 VAE 模型实例的 *引用* 传递给 Runner
            runner.set_vae_model(learner.vae_model) 
        else:
            logger.console_logger.error("Failed to link VAE: Learner (nq_vae_learner) or Runner (parallel_CFKeyAgent) mismatch.")
            logger.console_logger.error("确保 learner='nq_vae_learner' 并且 runner='parallel_CFKeyAgent'")
            # (如果失败，我们允许继续运行，但 VAE 将不起作用)
    # --- (新) ---

    # 是否读入已有权重
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        #检查 配置文件中的 测试or录像回放 项
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0#表示一个回合，即类似一局游戏，一次完整的任务尝试，智能体执行一系列动作与环境交互直到达到任务的终止条件,这里的episode表示当前这局游戏的序号
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:#训练时长未达到要求时长

        # Run for a whole episode at a time

        # 采集当前一局游戏的轨迹，存入经验回放池
        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)#ep*batch_run
        # 若轨迹数量支持采样一批batch，则训练
        if buffer.can_sample(args.batch_size):#这里的batch_size=n,则采样n条轨迹，共包含n*episode.max_t_filled()时间步
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)#采样batch_size条轨迹

            # Truncate batch to only filled timesteps
            # 数据处理，对齐，将所有轨迹的长度都变成相同（时间步=当前bs内最长那条轨迹的长度），便于神经网络处理，多出来的部分通过掩码mask掉
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]  #bs,ep

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            #调用learner开始训练，训练完成后释放一个episode的采样（buffer）
            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample



        #相关信息输出
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            #如果训练时间步超过了设定的测试时间点，进行n轮测试，以测试模式调用runner.run，并将结果输出到logger中
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            #如果save_model激活 并且 运行时间步 超出了save_model_interval，则保存模型
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #保存到results/models/ 下    "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            #如果运行时间步超过了设定的log时间点，则输出相关信息到logger中再打印到ternimal
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env


    #设定的训练时间步已到，结束运行（args.t_max）
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
