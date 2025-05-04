import pygame
from src.envs.bq_src.operator_env.op_Env import Op_Env
class Car_b_Env(Op_Env):
    def __init__(self, color, init_hex, id, type):
        super(Op_Env, self).__init__()
        self.color = color
        self.id = id
        self.type = type

        self.max_blood = 8
        self.cur_hex = init_hex
        self.cost_speed = 1
        self.can_be_seen = 10
        self.can_be_attack = 7
        self.attack_car = 1.5
        self.attack_car_pro = 0.7
        self.attack_peo = 0.8
        self.attack_peo_pro = 0.6

        self.shoot_cooling_time_cost = 1
        self.stop_can_shoot_time = 2

        self.blood = 8

        self.move_start_clock = -1
        self.move_time = 0
        self.stop_start_clock = -1
        self.stop_time = 0
        self.shoot_cooling_time = 0

        self.short_start_clock = -1

        self.can_see = []
        self.can_attack = []

        self.move_target_hex = init_hex

    def draw(self, screen):
        a = 5
        i = self.cur_hex // 100
        j = self.cur_hex % 100

        if i % 2 == 0:
            zhongxindian = ((2 * j + 1) * a, 2 * i * a + 1.5 * a)
        else:
            zhongxindian = ((j + 1) * 2 * a, 2 * a * i + 1.5 * a)

        if self.color == 0:
            bg_color = (250, 20, 120)
            tag = 0
        else:
            bg_color = (0, 100, 100)
            tag = 0

        b = 3
        rect1 = ((zhongxindian[0] - b / 2 + tag, zhongxindian[1] - b / 2 + tag), (b, b))
        pygame.draw.rect(screen, bg_color, rect1)
