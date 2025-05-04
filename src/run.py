import copy
import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
import numpy as np
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    print(_log)
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
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train

    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    # for coma
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


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    train_tasks = ["transbq1","transbq2"]
    ceshi_times, task2runner, task2args = {}, {}, {}
    task2scheme, task2groups, task2preprocess, task2buffer = {}, {}, {}, {}
    excel_filename = {}
    for task in train_tasks:
        excel_filename[task] = f'{task}.txt'
        # Set up schemes and groups here
        task_args = copy.deepcopy(args)
        if task == "transbq2":
            task_args.env_args["game_map_id"] = 3

        task2args[task] = task_args
        runner = r_REGISTRY[task_args.runner](args=task_args, logger=logger, task=task)
        task2runner[task] = runner
        env_info =runner.get_env_info()

        ceshi_times[task] = 0
        task_args.n_agents = env_info["n_agents"]
        task_args.n_actions = env_info["n_actions"]
        task_args.state_shape = env_info["state_shape"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.n_actions = env_info["n_actions"]
        scheme =  {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "avail_agents": {"vshape": (3,)}
        }

        # Default/Base scheme
        groups = {
            "agents": task_args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }
        buffer = ReplayBuffer(scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
                                     preprocess=preprocess,
                                     device="cpu" if task_args.buffer_cpu_only else task_args.device)

        task2buffer[task] = buffer
        task2scheme[task], task2groups[task], task2preprocess[task] = scheme, groups, preprocess

    task2buffer_scheme = {
        task: task2buffer[task].scheme for task in train_tasks
    }
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](train_tasks,task2buffer_scheme, task2groups, task2args, args)
    for task in train_tasks:
        # Give runner the scheme
        task2runner[task].setup(scheme=task2scheme[task], groups=task2groups[task], preprocess=task2preprocess[task], mac=mac)
    # Learner
    learner = le_REGISTRY[args.learner](mac, task2buffer_scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        raise Exception("We don't support checkpoint loading in multi-task learning currently!")
    # start training
    episode = 0
    model_save_time = 0  # 记录模型保存时间
    task2train_info = {task: {} for task in train_tasks}
    for task in train_tasks:
        task2train_info[task]["last_test_T"] = -args.test_interval - 1
        task2train_info[task]["last_log_T"] = 0
        task2train_info[task]["start_time"] = time.time()
        task2train_info[task]["last_time"] = task2train_info[task]["start_time"]
    logger.console_logger.info("Beginning multi-task training with {} timesteps for each task".format(args.t_max))

    task2terminated = {task: False for task in train_tasks}
    surrogate_task = np.random.choice(train_tasks)
    batch_size_train = args.batch_size
    batch_size_run = args.batch_size_run
    tot_run_time = {}
    task_start_time = {}
    for task_name in train_tasks:
        tot_run_time[task_name] = 0
    while True:
        if all(task2terminated.values()):
            # if all task learning terminated, jump out
            for task_name in train_tasks:
                logger.console_logger.info("[任务 {}] 耗时: {}".format(task_name, time_str(tot_run_time[task_name])))
            break
        # 每次训练前打乱任务顺序
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:
            task_start_time[task] = time.time()
            episode_batch = task2runner[task].run(test_mode=False)
            task2buffer[task].insert_episode_batch(episode_batch)
            if task2buffer[task].can_sample(batch_size_train):
                for _run in range(batch_size_run):
                    episode_sample = task2buffer[task].sample(batch_size_train)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train(episode_sample, task2runner[task].t_env, episode,task)

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // task2runner[task].batch_size)

            if (task2runner[task].t_env - task2train_info[task]["last_test_T"]) / args.test_interval >= 1.0:
                print('test begin time:', ceshi_times[task])
                # 打印一下各个任务的耗时
                # for task_name in train_tasks:
                logger.console_logger.info(
                    "[Task {}] t_env: {} / {}".format(task, task2runner[task].t_env, args.t_max))
                logger.console_logger.info("[Task {}] Estimated time left: {}. Time passed: {}".format(
                    task, time_left(task2train_info[task]["last_time"], task2train_info[task]["last_test_T"],
                                    task2runner[task].t_env, args.t_max),
                    time_str(time.time() - task2train_info[task]["start_time"])))
                task2train_info[task]["last_time"] = time.time()
                task2train_info[task]["last_test_T"] = task2runner[task].t_env
                win_tmp_ym = 0
                win_tmp_ym_2 = 0
                for _ in range(n_test_runs):
                    task2runner[task].run(test_mode=True)
                    if args.learner != 'coma_learner':
                        aly_blood_1, ene_blood_1 = task2runner[task].env.get_blood_score()
                        reward = aly_blood_1 - ene_blood_1
                        if reward > 0:
                            win_tmp_ym += 1
                        if reward > 0 and ene_blood_1 == 0:
                            win_tmp_ym_2 += 1
                print('test end time:', ceshi_times[task])
                ceshi_times[task] += 1
                if args.learner != 'coma_learner':
                    print('yao test times:', ceshi_times[task], 'ceshi win_rate',  win_tmp_ym / n_test_runs, 'ceshi win_rate2',  win_tmp_ym_2 / n_test_runs)
                    append_to_txt(excel_filename[task], win_tmp_ym / n_test_runs)

        if args.save_model and task == surrogate_task and (task2runner[task].t_env - model_save_time >= args.save_model_interval or model_save_time== 0):
            model_save_time = task2runner[task].t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(task2runner[task].t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (task2runner[task].t_env - task2train_info[task]["last_test_T"]) >= args.log_interval:
            logger.log_stat("episode", episode, task2runner[task].t_env)
            logger.print_recent_stats()
            task2train_info[task]["last_test_T"] = task2runner[task].t_env

        if task2runner[task].t_env > args.t_max:  # 如果当前环境步数t_env超过最大步数t_max,则标记task已结束
            task2terminated[task] = True
            # schedule surrogate task
            if task == surrogate_task and not all(task2terminated.values()):
                surrogate_task = np.random.choice([task for task in train_tasks if not task2terminated[task]])
                model_save_time = -1
    # runner.close_env()
    # logger.console_logger.info("Finished Training")


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


def append_to_txt(txt_filename, win_rate):
    try:
        # 检查文件是否存在
        if not os.path.exists(txt_filename):
            # 如果文件不存在，创建新文件并写入表头
            with open(txt_filename, 'w') as file:
                file.write("number\twin_rate\n")
            new_index = 1
        else:
            # 如果文件存在，读取最后一行获取最大序号
            with open(txt_filename, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    new_index = int(last_line.split('\t')[0]) + 1
                else:
                    new_index = 1

        # 添加新数据
        with open(txt_filename, 'a') as file:
            file.write(f"{new_index}\t{win_rate:.4f}\n")

        print(f"数据已成功添加到 {txt_filename}")
    except Exception as e:
        print(f"发生错误: {str(e)}")