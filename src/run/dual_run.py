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

def run_dual(args, logger):
    """
    Dual Run 入口函数：
    既支持正常的双重训练循环，也支持通过 --evaluate 参数仅进行模型评估。
    """
    
    # =============================================================================
    # 1. 初始化部分 (Runner & Env Info)
    # =============================================================================
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # =============================================================================
    # 2. 评估模式检查 (Evaluate Mode)
    #    如果设置了 evaluate=True，则转入评估逻辑，不再进行训练
    # =============================================================================
    if args.evaluate:
        _evaluate_dual(args, runner, logger)
        return

    # =============================================================================
    # 3. 正常训练流程 (Training Mode)
    # =============================================================================
    
    # Default/Base scheme
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

    # Replay Buffer
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else "cuda")

    # Setup MAC
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Learner 初始化 (Dual Learner)
    # (A) 初始化 World Model Learner
    if not hasattr(args, "wm_batch_size"):
        args.wm_batch_size = args.batch_size
        
    wm_learner = WMLearner(args)
    if args.use_cuda:
        wm_learner.cuda()

    # (B) 初始化 RL Learner (注入 WM)
    rl_learner = DVDWMCausalLearner(mac, buffer.scheme, logger, args, wm_learner=wm_learner)
    if args.use_cuda:
        rl_learner.cuda()

    # (C) 加载 Checkpoint (如果存在)
    if args.checkpoint_path != "":
        _load_checkpoint(args, rl_learner, wm_learner, logger)
        # 如果加载了 checkpoint，通常将 t_env 更新为加载的步数
        # 这里需要从 path 解析出 timestep，逻辑封装在 _load_checkpoint 比较复杂
        # 简单起见，我们假设用户知道自己在做什么，或者在 _load_checkpoint 里处理 runner.t_env
        pass 

    # -----------------------------------------------------------------------------
    # 训练主循环
    # -----------------------------------------------------------------------------
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning dual training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # Dual Sampling Logic
        # Dual Sampling Logic
        # 计算两者中最大的 Batch Size
        max_batch_needed = max(args.batch_size, args.wm_batch_size)

        if buffer.can_sample(max_batch_needed):
            
            # 1. 统一采样 (Sample Once)
            # 采样足够大的 Batch 以满足两者需求
            episode_sample = buffer.sample(max_batch_needed)
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 2. 训练 World Model
            # 如果 wm_bs 等于 max，直接用；否则切取前 wm_bs 个数据
            if args.wm_batch_size == max_batch_needed:
                wm_batch = episode_sample
            else:
                wm_batch = episode_sample[:args.wm_batch_size]
            
            wm_learner.train(wm_batch, runner.t_env, logger)
            
            # 3. 训练 RL Agent
            # 如果 rl_bs 等于 max，直接用；否则切取前 rl_bs 个数据
            # 关键点：这里使用的是和 WM 相同来源的数据（子集）
            if args.batch_size == max_batch_needed:
                rl_batch = episode_sample
            else:
                rl_batch = episode_sample[:args.batch_size]
                
            rl_learner.train(rl_batch, runner.t_env, episode)
            
            # 显式删除引用，辅助 GC
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # Save models
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
    """
    专门用于评估的函数 (对应原生的 evaluate_sequential)
    """
    logger.console_logger.info("Running evaluation (Dual Mode)...")
    
    # 必须初始化 MAC 和 Learner 才能加载权重
    # 虽然评估主要靠 MAC，但 DVDWMCausalLearner 的结构可能比较复杂，
    # 且 args.checkpoint_path 需要被正确加载
    
    # 1. 建立临时的 Buffer Scheme (为了初始化 MAC)
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
    
    # 2. Init MAC
    mac = mac_REGISTRY[args.mac](scheme, groups, args)
    
    # 3. Init Learners (为了加载模型参数)
    if not hasattr(args, "wm_batch_size"):
        args.wm_batch_size = args.batch_size
    wm_learner = WMLearner(args)
    if args.use_cuda: wm_learner.cuda()
    
    rl_learner = DVDWMCausalLearner(mac, scheme, logger, args, wm_learner=wm_learner)
    if args.use_cuda: rl_learner.cuda()

    # 4. 加载模型
    if args.checkpoint_path != "":
        _load_checkpoint(args, rl_learner, wm_learner, logger)
        runner.t_env = 0 # 评估模式下 t_env 重置或从 checkpoint 读取皆可，主要影响 log
    else:
        logger.console_logger.warning("Evaluate mode is on but no checkpoint_path provided! Testing with random weights.")

    # 5. 执行评估循环
    # 运行 args.test_nepisode 次
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    
    logger.console_logger.info(f"Starting evaluation for {args.test_nepisode} episodes...")
    for _ in range(n_test_runs):
        runner.run(test_mode=True)
        
    logger.print_recent_stats()
    runner.close()
    logger.console_logger.info("Finished Evaluation")


def _load_checkpoint(args, rl_learner, wm_learner, logger):
    """
    辅助函数：加载 Checkpoint
    """
    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info("Checkpoint direct path: {}".format(args.checkpoint_path))
        rl_learner.load_models(args.checkpoint_path)
        try:
            wm_learner.load_models(args.checkpoint_path)
        except:
            logger.console_logger.warning("Could not load World Model from checkpoint (maybe not saved or path error)!")
        return

    # 标准目录结构加载
    logger.console_logger.info("Loading checkpoint from {}".format(args.checkpoint_path))
    for name in os.listdir(args.checkpoint_path):
        full_name = os.path.join(args.checkpoint_path, name)
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if args.load_step == 0:
        timestep_to_load = max(timesteps)
    else:
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    rl_learner.load_models(model_path)
    # 尝试加载 WM，如果不强制要求 WM 必须存在（比如只想测 RL），可以加 try-except
    # 但由于我们的架构强依赖 WM，这里抛出异常是合理的
    try:
        wm_learner.load_models(model_path)
    except Exception as e:
         logger.console_logger.warning(f"Failed to load World Model: {e}. Check if wm_opt.th/world_model.th exists.")