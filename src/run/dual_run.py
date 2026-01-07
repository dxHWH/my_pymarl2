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

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

# 引入我们的自定义 Learner
from learners.wm_learner import WMLearner
from learners.dvd_wm_causal_learner import DVDWMCausalLearner

from learners.dvd_wm_fac_learner import DVDWMFacLearner


def run_dual(_run, config, _log):
    """
    Dual Run 入口函数
    """

    # 将 config 字典转换为 Namespace (args)
    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"
    
    # 封装 Logger
    logger = Logger(_log)

    # =============================================================================
    # [修复 1] 配置 Tensorboard 和 Sacred (参照原生 run.py)
    # =============================================================================
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 生成 unique_token
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    
    # 配置 Tensorboard Logger
    if args.use_tensorboard:
        # 路径结构: results/tb_logs/map_name/unique_token
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs", args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # =============================================================================
    # 1. 初始化部分
    # =============================================================================
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # =============================================================================
    # 2. 评估模式检查
    # =============================================================================
    if args.evaluate:
        _evaluate_dual(args, runner, logger)
        return

    # =============================================================================
    # 3. 正常训练流程
    # =============================================================================
    
    # Scheme Definition
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else "cuda")

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Setup Runner
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner Init
    if not hasattr(args, "wm_batch_size"):
        args.wm_batch_size = args.batch_size
        
    wm_learner = WMLearner(args)
    if args.use_cuda:
        wm_learner.cuda()

    rl_learner = DVDWMCausalLearner(mac, buffer.scheme, logger, args, wm_learner=wm_learner)
    if args.use_cuda:
        rl_learner.cuda()

    if args.checkpoint_path != "":
        _load_checkpoint(args, rl_learner, wm_learner, logger)

    # -----------------------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------------------
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning dual training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # 1. Collect Data
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # 2. Train (Dual Sampling)
        max_batch_needed = max(args.batch_size, args.wm_batch_size)

        if buffer.can_sample(max_batch_needed):
            episode_sample = buffer.sample(max_batch_needed)
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # Train WM
            if args.wm_batch_size == max_batch_needed:
                wm_batch = episode_sample
            else:
                wm_batch = episode_sample[:args.wm_batch_size]
            wm_learner.train(wm_batch, runner.t_env, logger)
            
            # Train RL
            if args.batch_size == max_batch_needed:
                rl_batch = episode_sample
            else:
                rl_batch = episode_sample[:args.batch_size]
            rl_learner.train(rl_batch, runner.t_env, episode)
            
            del episode_sample

        # 3. Test & Evaluate
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            
            # [修复 2] 运行测试并评估 WM 性能
            for i in range(n_test_runs):
                test_batch = runner.run(test_mode=True)
                # 只用第一个 batch 评估 WM，避免计算太慢
                if i == 0:
                    if test_batch.device != args.device:
                        test_batch.to(args.device)
                    wm_learner.evaluate(test_batch, runner.t_env, logger)

        # 4. Save Models
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            rl_learner.save_models(save_path)
            wm_learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close()
    logger.console_logger.info("Finished Training")


def _evaluate_dual(args, runner, logger):
    logger.console_logger.info("Running evaluation (Dual Mode)...")
    # ... (evaluation logic mostly same as before) ...
    # 为了节省篇幅，这里略去重复的 Setup 代码，
    # 但请确保这里的 setup 逻辑与 run_dual 开头一致 (Setup runner/mac/learners)
    # 并且如果在 evaluate 模式下也想看 tensorboard，需要在外面就配好
    # (上面的 run_dual 入口函数已经配好了)
    
    # 简化的 setup 流程复刻:
    env_info = runner.get_env_info()
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    mac = mac_REGISTRY[args.mac](scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    if not hasattr(args, "wm_batch_size"): args.wm_batch_size = args.batch_size
    wm_learner = WMLearner(args)
    if args.use_cuda: wm_learner.cuda()
    rl_learner = DVDWMCausalLearner(mac, scheme, logger, args, wm_learner=wm_learner)
    if args.use_cuda: rl_learner.cuda()

    if args.checkpoint_path != "":
        _load_checkpoint(args, rl_learner, wm_learner, logger)
        runner.t_env = 0 
    
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    logger.console_logger.info(f"Starting evaluation for {args.test_nepisode} episodes...")
    for i in range(n_test_runs):
        test_batch = runner.run(test_mode=True)
        if i == 0:
             if test_batch.device != args.device: test_batch.to(args.device)
             wm_learner.evaluate(test_batch, runner.t_env, logger)
        
    logger.print_recent_stats()
    runner.close()
    logger.console_logger.info("Finished Evaluation")

def _load_checkpoint(args, rl_learner, wm_learner, logger):
    # ... (保持不变) ...
    timesteps = []
    timestep_to_load = 0
    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info("Checkpoint direct path: {}".format(args.checkpoint_path))
        rl_learner.load_models(args.checkpoint_path)
        try: wm_learner.load_models(args.checkpoint_path)
        except: pass
        return
    for name in os.listdir(args.checkpoint_path):
        full_name = os.path.join(args.checkpoint_path, name)
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))
    if args.load_step == 0: timestep_to_load = max(timesteps)
    else: timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))
    model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
    logger.console_logger.info("Loading model from {}".format(model_path))
    rl_learner.load_models(model_path)
    try: wm_learner.load_models(model_path)
    except Exception as e: logger.console_logger.warning(f"Failed to load WM: {e}")