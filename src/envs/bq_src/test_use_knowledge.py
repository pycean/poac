from src.envs.bq_src.war_gym import War_Env
import pygame
from pygame import time as tm
from pygame.locals import *
from copy import deepcopy
from src.learners import REGISTRY as le_REGISTRY
from src.runners import REGISTRY as r_REGISTRY
from src.controllers import REGISTRY as mac_REGISTRY
from src.envs import REGISTRY as env_REGISTRY
from src.components.transforms import OneHot
from src.components.episode_buffer import ReplayBuffer
from src.components.episode_buffer import EpisodeBatch
from types import SimpleNamespace as SN
import torch as th
import json
from functools import partial
from src.utils.logging import Logger


class EpisodeRunner:

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

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def run(self, test_mode=False):

        for i_episode in range(10):
            self.reset()
            for t in range(1000):
                terminated = False
                episode_return = 0
                self.mac.init_hidden(batch_size=args.batch_size_run)
                while not terminated:
                    self.env.render(screen)
                    pygame.display.flip()
                    tm.wait(1000)
                    pre_transition_data = {
                        "state": [self.env.get_state()],
                        "avail_actions": [self.env.get_avail_actions()],
                        "obs": [self.env.get_obs()],
                        "avail_agents": [self.env.get_valid_agent()]
                    }

                    self.batch.update(pre_transition_data, ts=self.t)

                    # Pass the entire batch of experiences up till now to the agents
                    # Receive the actions for each agent at this timestep in a batch of size 1
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

                    # action = [int(a) for a in actions[0]]
                    print(actions[0])
                    reward, terminated, env_info = self.env.step(actions[0])
                    episode_return += reward
                    post_transition_data = {
                        "actions": actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }

                    self.batch.update(post_transition_data, ts=self.t)
                #
                    self.t += 1
                    #
                    # last_data = {
                    #     "state": [self.env.get_state()],
                    #     "avail_actions": [self.env.get_avail_actions()],
                    #     "obs": [self.env.get_obs()],
                    #     "avail_agents": [self.env.get_valid_agent()]
                    # }
                    # self.batch.update(last_data, ts=self.t)
                    #
                    # # Select actions in the last stored state
                    # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                    #
                    # self.batch.update({"actions": actions}, ts=self.t)
                    #
                    # cur_stats = self.test_stats if test_mode else self.train_stats
                    # cur_returns = self.test_returns if test_mode else self.train_returns
                    # log_prefix = "test_" if test_mode else ""
                    # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
                    # cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
                    # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
                    #
                    if not test_mode:
                        print('train step', self.t, 'train episode_return', episode_return)
                #
                # cur_returns.append(episode_return)
                #
                # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
                #     pass
                # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                #     pass
                #     if hasattr(self.mac.action_selector, "epsilon"):
                #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                #     self.log_train_stats_t = self.t_env

                if terminated:
                    print("Episode finished after {} timesteps".format(t + 1))
                    print('i_episode', i_episode)
                    break

        pygame.quit()

    # # Demonstration loop
    # while keep_going:
    #     runner.run(test_mode=False)
    #
    #     # If you need additional statistics during demonstration
    #     if args.learner != 'coma_learner':
    #         aly_blood_1, ene_blood_1 = runner.env.get_blood_score()
    #         reward = aly_blood_1 - ene_blood_1
    #         episode += 1
    #
    #     for event in pygame.event.get():
    #         if event.type == QUIT:
    #             keep_going = False
    #     runner.env.render(screen)
    #     pygame.display.flip()
    #     clock.tick(60)
    #
    #     if episode >= n_test_runs:
    #         keep_going = False
    #
    # runner.close_env()
    # pygame.quit()

if __name__ == '__main__':
    file_path = r'E:\wyh\POAC-pymarl\results\sacred\5\config.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    _config = deepcopy(config)
    args = SN(**_config)

    _log = '<Logger my_main (DEBUG)>'
    logger = Logger(_log)
    env = War_Env(0, 0, 0)
    pygame.init()
    size = width, height = 1000, 600
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("bq游戏")
    clock = pygame.time.Clock()

    bg_color = (200, 200, 200)

    screen.fill(bg_color)

    env_info = env.get_env_info()

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.device = 'cpu'

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "avail_agents": {"vshape": (3,)}
    }

    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    learner.load_models(r'E:\wyh\POAC-pymarl\results\models\qmix__2024-08-07_19-54-51\8763570')
    runner = EpisodeRunner(args=args, logger=logger)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    runner.run(test_mode=False)
    #
    # keep_going = True
    #
    #
    # # 加载模型参数
    # learner.load_models(r'E:\wyh\POAC-pymarl\results\models\qmix__2024-08-10_11-33-04\154285')
    #
    # def play_game():
    #      for i_episode in range(10):
    #         env.reset()
    #         # operators_ob, state = env.reset()
    #         for t in range(1000):
    #             env.render(screen)
    #             pygame.display.flip()
    #             tm.wait(1000)
    #             batch = partial(EpisodeBatch, scheme, groups, 1, 601, preprocess=preprocess, device=args.device)
    #             pre_transition_data = {
    #                 "state": [env.get_state()],
    #                 "avail_actions": [env.get_avail_actions()],
    #                 "obs": [env.get_obs()],
    #                 "avail_agents": [env.get_valid_agent()]
    #             }
    #             t = 0
    #             t_env = 600
    #             test_mode = False
    #
    #             batch.update(pre_transition_data, ts=t)
    #             actions = mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=test_mode)
    #             reward, done, env_info = env.step(actions[0])
    #             post_transition_data = {
    #                 "actions": actions,
    #                 "reward": [(reward,)],
    #                 "terminated": [(terminated != env_info.get("episode_limit", False),)],
    #             }
    #
    #             batch.update(post_transition_data, ts=t)
    #             t += 1
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
    #
    # pygame.quit()