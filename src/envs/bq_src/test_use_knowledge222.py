from src.envs.bq_src.war_gym import War_Env
import pygame
from pygame import time as tm
from pygame.locals import *

env = War_Env(0,0,0)
pygame.init()
size = width, height = 1000, 600
screen = pygame.display.set_mode(size)
pygame.display.set_caption("bq游戏")
clock = pygame.time.Clock()

keep_going = False

bg_color = (200, 200, 200)

screen.fill(bg_color)

for i_episode in range(10):
    operators_ob, actions_valid = env.reset()
    for t in range(1000):
        env.render(screen)
        pygame.display.flip()
        tm.wait(1000)
        action = env.action_space_sample(0)
        operators_ob, done, actions_valid = env.step(action )

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print('i_episode', i_episode)
            break

# for i_episode in range(10):
#     operators_ob, actions_valid, score_ob = env.reset()
#     for t in range(1000):
#         env.render(screen)
#         pygame.display.flip()
#         tm.wait(1000)
#         action = env.action_space_sample()
#         operators_ob, actions_valid, score_ob, clock1, done = env.step(action)
#         print('clock', clock1)
#
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             print('i_episode', i_episode)
#             break

pygame.quit()

