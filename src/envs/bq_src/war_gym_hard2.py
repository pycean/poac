
import json
from src.envs.bq_src.operator_env.car_A_Env import Car_a_Env
from src.envs.bq_src.operator_env.car_B_Env import Car_b_Env
from src.envs.bq_src.operator_env.people_Env import People_Env
from src.envs.bq_src.map.map import Map
from src.envs.multiagentenv import MultiAgentEnv
import numpy as np
import pygame
import torch as th
from pygame.locals import *

class War_Env_hard2(MultiAgentEnv):
    def __init__(self, game_agent_id, game_map_id, owner, state_last_action=False):
        print("this is bq hard 2 env")
        self.game_id = game_agent_id
        self.game_map_id = game_map_id
        self.episode_limit = 600
        self.owner = owner

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.win_n = 0
        self.win_l = 0
        self.force_restarts = 0

        self.guide_attack_value = 2
        self.guide_attack_pro = 1
        self.guide_attack_dis = 9

        self.obs_instead_of_state = False

        with open("src/envs/bq_src/" + str(self.game_id) + ".json", 'r') as load_f:
            self.game_dict = json.load(load_f)

        self.agents_alive_list = [{}, {}]
        self.agentid_index_dict = [{}, {}]

        self.index_agentid_dict = [{}, {}]
        self.agentid_type_dict = {}
        self.agentid_color_dict = {}
        self.load_agents()

        self.map = Map(self.game_map_id)

        self.clock = 0
        self.n_agents = 3

        self.win_counted = False

        self.guide_num =0

        self.use_invalid = False
        if self.use_invalid:
            # move 0-5 shoot 6-8 guide 9 10 11 stop 12  invalid 13
            self.n_actions = 11 + 3
        else:
            self.n_actions = 10 + 3
        print('env is ready')

    def load_agents(self):
        index_0 = 0
        index_1 = 0
        for agent_dict in self.game_dict['operators']:
            if agent_dict['type'] == 0:
                agent_instance = Car_a_Env(agent_dict['color'], agent_dict['init_hex'], agent_dict['id'],
                                           agent_dict['type'])
            elif agent_dict['type'] == 1:
                agent_instance = Car_b_Env(agent_dict['color'], agent_dict['init_hex'], agent_dict['id'],
                                           agent_dict['type'])
            elif agent_dict['type'] == 2:
                agent_instance = People_Env(agent_dict['color'], agent_dict['init_hex'], agent_dict['id'],
                                           agent_dict['type'])
            else:
                raise
            self.agents_alive_list[agent_dict['color']][agent_dict['id']] = agent_instance
            self.agentid_color_dict[agent_dict['id']] = agent_dict['color']
            self.agentid_type_dict[agent_dict['id']] = agent_dict['type']
            if agent_dict['color'] == 0:
                self.agentid_index_dict[agent_dict['color']][agent_dict['id']] = index_0
                self.index_agentid_dict[agent_dict['color']][index_0] = agent_dict['id']
                index_0 += 1
            else:
                self.agentid_index_dict[agent_dict['color']][agent_dict['id']] = index_1
                self.index_agentid_dict[agent_dict['color']][index_1] = agent_dict['id']
                index_1 += 1

    def update_agents_alive_list(self):
        for color in [0, 1]:
            for agent_id in self.agents_alive_list[color].keys():
                if self.agents_alive_list[color][agent_id] != None:
                    if self.agents_alive_list[color][agent_id].blood <= 0:
                        self.agents_alive_list[color][agent_id] = None

    def update_agent_see_attack(self):
        for agent_id in self.agents_alive_list[1].keys():
            if self.agents_alive_list[1][agent_id] != None:
                self.agents_alive_list[1][agent_id].can_see = []
                self.agents_alive_list[1][agent_id].can_attack = []

        for agent_id in self.agents_alive_list[0].keys():
            if self.agents_alive_list[0][agent_id] != None:
                self.agents_alive_list[0][agent_id].can_see = []
                self.agents_alive_list[0][agent_id].can_attack = []
                for enemy_id in self.agents_alive_list[1].keys():
                    if self.agents_alive_list[1][enemy_id] != None:
                        distance = self.map.get_dis_between_hex(
                            self.agents_alive_list[0][agent_id].cur_hex,
                            self.agents_alive_list[1][enemy_id].cur_hex
                        )
                        agent_can_see = self.agents_alive_list[0][agent_id].can_be_seen
                        agent_cur_hex = self.agents_alive_list[0][agent_id].cur_hex
                        if self.map.map_data[agent_cur_hex//100][agent_cur_hex%100][0] == 1.0:
                            agent_can_see = agent_can_see / 2
                        enemy_can_see = self.agents_alive_list[1][enemy_id].can_be_seen
                        enemy_cur_hex = self.agents_alive_list[1][enemy_id].cur_hex
                        if self.map.map_data[enemy_cur_hex//100][enemy_cur_hex%100][0] == 1.0:
                            enemy_can_see = enemy_can_see / 2
                        if distance <= agent_can_see:
                            self.agents_alive_list[1][enemy_id].can_see.append(agent_id)
                            if distance <= self.agents_alive_list[0][agent_id].can_be_attack:
                                self.agents_alive_list[1][enemy_id].can_attack.append(agent_id)
                        if distance <= enemy_can_see:
                            self.agents_alive_list[0][agent_id].can_see.append(enemy_id)
                            if distance <= self.agents_alive_list[1][enemy_id].can_be_attack:
                                self.agents_alive_list[0][agent_id].can_attack.append(enemy_id)

    def get_obs_agent(self, agent_id):
        feature_list = [np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32), ]
        agent = None
        can_see_list = []
        color = self.agentid_color_dict[agent_id]
        if agent_id in self.agents_alive_list[color].keys():
            agent = self.agents_alive_list[color][agent_id]
        else:
            raise
        if agent:
            can_see_list = agent.can_see
            for id in self.agents_alive_list[color]:
                index_agent = self.agentid_index_dict[color][id]
                if self.agents_alive_list[color][id] == None:
                    continue
                feature_list[index_agent][0] = self.agents_alive_list[color][id].color
                feature_list[index_agent][1] = self.agents_alive_list[color][id].id / 10.0
                feature_list[index_agent][2] = self.agents_alive_list[color][id].type /5.0
                feature_list[index_agent][3] = self.agents_alive_list[color][id].cur_hex / 1500.0
                feature_list[index_agent][4] = self.agents_alive_list[color][id].blood /20.0
                feature_list[index_agent][5] = self.agents_alive_list[color][id].move_time /100.0
                feature_list[index_agent][6] = self.agents_alive_list[color][id].stop_time /100.0
                feature_list[index_agent][7] = self.agents_alive_list[color][id].shoot_cooling_time /100.0
                can_see_index = 8
                for ene_id in self.agents_alive_list[color][id].can_see:
                    feature_list[index_agent][can_see_index + self.agentid_index_dict[1-color][ene_id]] = 1.0
                can_attack_index = 11
                for ene_id in self.agents_alive_list[color][id].can_attack:
                    feature_list[index_agent][can_attack_index + self.agentid_index_dict[1 - color][ene_id]] = 1.0
                # feature_list[index_agent][14] = self.clock / 600.0

            for id in self.agents_alive_list[1-color]:
                if id in can_see_list:
                    index_agent = self.agentid_index_dict[1-color][id] + 3
                    if self.agents_alive_list[1-color][id] == None:
                        continue
                    feature_list[index_agent][0] = self.agents_alive_list[1-color][id].color
                    feature_list[index_agent][1] = self.agents_alive_list[1-color][id].id /10.0
                    feature_list[index_agent][2] = self.agents_alive_list[1-color][id].type / 5.0
                    feature_list[index_agent][3] = self.agents_alive_list[1-color][id].cur_hex / 1500.0
                    feature_list[index_agent][4] = self.agents_alive_list[1-color][id].blood / 20.0
                    # feature_list[index_agent][14] = self.clock / 600.0

        agent_ind = self.agentid_index_dict[color][agent_id]
        sort_feature_list = []
        own_feature = feature_list[agent_ind]
        sort_feature_list.append(own_feature)
        for i in range(len(feature_list)):
            if i != agent_ind:
                sort_feature_list.append(feature_list[i])

        agent_obs = np.concatenate(
            (
                sort_feature_list[0].flatten(),
                sort_feature_list[1].flatten(),
                sort_feature_list[2].flatten(),
                sort_feature_list[3].flatten(),
                sort_feature_list[4].flatten(),
                sort_feature_list[5].flatten(),
            )
        )
        agent_obs = np.append(
            agent_obs, self.clock / 600.0
        )
        return agent_obs

    def get_obs(self):
        agent_obs = [[] for i in range(self.n_agents)]
        for agent_id in self.agentid_index_dict[self.owner].keys():
            agent_obs[self.agentid_index_dict[self.owner][agent_id]] = self.get_obs_agent(agent_id)
        return agent_obs

    def get_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
        feature_list = [np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32), np.zeros(14, dtype=np.float32),
                        np.zeros(14, dtype=np.float32), ]

        for color in [0, 1]:
            for id in self.agents_alive_list[color]:
                index_agent = self.agentid_index_dict[color][id] + color * 3
                if self.agents_alive_list[color][id] == None:
                    continue
                feature_list[index_agent][0] = self.agents_alive_list[color][id].color
                feature_list[index_agent][1] = self.agents_alive_list[color][id].id / 10.0
                feature_list[index_agent][2] = self.agents_alive_list[color][id].type / 5.0
                feature_list[index_agent][3] = self.agents_alive_list[color][id].cur_hex / 1500.0
                feature_list[index_agent][4] = self.agents_alive_list[color][id].blood / 20.0
                feature_list[index_agent][5] = self.agents_alive_list[color][id].move_time / 100.0
                feature_list[index_agent][6] = self.agents_alive_list[color][id].stop_time / 100.0
                feature_list[index_agent][7] = self.agents_alive_list[color][id].shoot_cooling_time / 100.0
                can_see_index = 8
                for ene_id in self.agents_alive_list[color][id].can_see:
                    feature_list[index_agent][can_see_index + self.agentid_index_dict[1 - color][ene_id]] = 1.0
                can_attack_index = 11
                for ene_id in self.agents_alive_list[color][id].can_attack:
                    feature_list[index_agent][can_attack_index + self.agentid_index_dict[1 - color][ene_id]] = 1.0

        state = np.concatenate(
            (
                feature_list[0].flatten(),
                feature_list[1].flatten(),
                feature_list[2].flatten(),
                feature_list[3].flatten(),
                feature_list[4].flatten(),
                feature_list[5].flatten(),
            )
        )
        state = np.append(
            state, self.clock / 600.0
        )
        return state

    def get_total_actions(self):
        return self.n_actions

    def get_avail_guide_shoot(self):
        guide_list = [0, 0, 0]
        peo_guide = None
        car_guide = None
        for agent_id in self.agents_alive_list[self.owner].keys():
            if self.agents_alive_list[self.owner][agent_id] != None:
                if self.agents_alive_list[self.owner][agent_id].type == 1:
                    car_guide = self.agents_alive_list[self.owner][agent_id]
                if self.agents_alive_list[self.owner][agent_id].type == 2:
                    peo_guide = self.agents_alive_list[self.owner][agent_id]

        if peo_guide != None and car_guide != None:
            if peo_guide.can_see != []:
                if peo_guide.shoot_cooling_time >= peo_guide.shoot_cooling_time_cost \
                        and peo_guide.stop_time >= peo_guide.stop_can_shoot_time:
                    if car_guide.shoot_cooling_time >= car_guide.shoot_cooling_time_cost \
                            and car_guide.stop_time >= car_guide.stop_can_shoot_time:
                        for en_id in peo_guide.can_see:
                            en_instance = self.agents_alive_list[1-self.owner][en_id]
                            if en_instance.type == 0 or en_instance.type == 1:
                                ene_index = self.agentid_index_dict[1-self.owner][en_id]

                                dis = self.map.get_dis_between_hex(en_instance.cur_hex, car_guide.cur_hex)
                                if dis <= self.guide_attack_dis:
                                    guide_list[ene_index] = 1

        return guide_list



    def get_avail_actions(self):
        avail_actions = [[] for i in range(self.n_agents)]
        for agent_id in self.agentid_index_dict[self.owner].keys():
            avail_actions[self.agentid_index_dict[self.owner][agent_id]] = self.get_avail_agent_actions(agent_id)
        print('yao', avail_actions)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):

        color = self.agentid_color_dict[agent_id]
        avail_actions = [0] * self.n_actions

        # non guide
        if self.use_invalid:
            avail_actions[-1] = 1
        if agent_id in self.agents_alive_list[color].keys():
            agent = self.agents_alive_list[color][agent_id]
            if agent == None:
                avail_actions = [1] * self.n_actions
                avail_actions[9] = 0
                avail_actions[10] = 0
                avail_actions[11] = 0
                return avail_actions
            action_list = agent.execute_before_all_actors(self.map)
            for i in action_list[0]:
                avail_actions[i] = 1
                if self.use_invalid:
                    avail_actions[-2] = 1
                else:
                    avail_actions[-1] = 1
            for attack_id in action_list[1]:
                index = self.agentid_index_dict[1-color][attack_id] + 6
                avail_actions[index] = 1
        # guide type 1 zhanche
        if self.agentid_type_dict[agent_id] == 1:
            guide_list = self.get_avail_guide_shoot()
            for index in range(len(guide_list)):
                if guide_list[index] == 1:
                    avail_actions[index + 9] = 1
        if sum(avail_actions) == 0:
            avail_actions = [1] * self.n_actions
            avail_actions[9] = 0
            avail_actions[10] = 0
            avail_actions[11] = 0
        return avail_actions

    def get_avail_agent_actions_fact(self, agent_id):
        color = self.agentid_color_dict[agent_id]
        avail_actions = [0] * self.n_actions

        # non guide
        if self.use_invalid:
            avail_actions[-1] = 1
        if agent_id in self.agents_alive_list[color].keys():
            agent = self.agents_alive_list[color][agent_id]
            if agent == None:
                return avail_actions
            action_list = agent.execute_before_all_actors(self.map)
            for i in action_list[0]:
                avail_actions[i] = 1
                if self.use_invalid:
                    avail_actions[-2] = 1
                else:
                    avail_actions[-1] = 1
            for attack_id in action_list[1]:
                index = self.agentid_index_dict[1 - color][attack_id] + 6
                avail_actions[index] = 1
        # guide
        if self.agentid_type_dict[agent_id] == 1:
            guide_list = self.get_avail_guide_shoot()
            for index in range(len(guide_list)):
                if guide_list[index] == 1:
                    avail_actions[index + 9] = 1

        return avail_actions

    def get_valid_agent(self,):
        vaild_agent_list = [1, 1, 1]
        for agent_id in self.agents_alive_list[self.owner].keys():
            avail_actions = self.get_avail_agent_actions_fact(agent_id)
            if sum(avail_actions) == 0:
                ind = self.agentid_index_dict[self.owner][agent_id]
                vaild_agent_list[ind] = 0
        return vaild_agent_list


    def reset(self):
        self.clock = 0
        self.load_agents()
        self.update_agent_see_attack()
        self.win_counted = False
        self.guide_num = 0
        return self.get_obs(), self.get_state()

    def is_done(self):
        done_0 = True
        for agent_id in self.agents_alive_list[0].keys():
            if self.agents_alive_list[0][agent_id] != None:
                done_0 = False
        done_1 = True
        for agent_id in self.agents_alive_list[1].keys():
            if self.agents_alive_list[1][agent_id] != None:
                done_1 = False
        if done_0 or done_1:
            return True, [done_0, done_1]
        else:
            return False, [done_0, done_1]


    def action_space_sample(self, knAI_color):
        import numpy as np
        if self.game_map_id == 0:
            map_dis_for_ai = 6
        else:
            map_dis_for_ai = 7
        mid_point = (self.map.map_size[self.game_map_id][0] - 1) / 2 * 100 + (
                    self.map.map_size[self.game_map_id][1] - 1) / 2
        mid_point = int(mid_point)
        actions_dict = {}
        for ene_id in self.agents_alive_list[knAI_color]:
            ene_agent = self.agents_alive_list[knAI_color][ene_id]
            if ene_agent:
                actions = ene_agent.execute_before_all_actors(self.map)
                move_dis_dict = {}
                sorted_move_list = []
                attack_type_dict = {}
                for move_tmp in actions[0]:
                    hex = self.map.get_spec_len_dir_pos(ene_agent.cur_hex, 1, move_tmp)
                    dis = self.map.get_dis_between_hex(hex, mid_point)
                    move_dis_dict[move_tmp] = dis
                if move_dis_dict != {}:
                    sorted_move_list = sorted(move_dis_dict.items(), key=lambda item: item[1], reverse=False)

                for attack_tmp in actions[1]:
                    attack_type_dict[self.agentid_type_dict[attack_tmp]] = attack_tmp

                if actions[1] != []:
                    if 0 in attack_type_dict.keys():
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[0]] + 6
                    elif 1 in attack_type_dict.keys():
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[1]] + 6
                    else:
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[2]] + 6
                else:
                    if sorted_move_list != []:
                        if ene_agent.type != 0:
                            if ene_agent.can_attack == []:
                                if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai:
                                    actions_dict[ene_id] = sorted_move_list[0][0]
                                else:
                                    actions_dict[ene_id] = np.random.choice(actions[0])
                        else:
                            if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai:
                                actions_dict[ene_id] = sorted_move_list[0][0]
                            else:
                                actions_dict[ene_id] = np.random.choice(actions[0])
        return actions_dict

    def action_space_sample_AI1(self, knAI_color):
        import numpy as np
        if self.game_map_id == 0:
            map_dis_for_ai = 6
        else:
            map_dis_for_ai = 7
        mid_point = (self.map.map_size[self.game_map_id][0] - 1) / 2 * 100 + (
                    self.map.map_size[self.game_map_id][1] - 1) / 2
        mid_point = int(mid_point)
        actions_dict = {}
        for ene_id in self.agents_alive_list[knAI_color]:
            ene_agent = self.agents_alive_list[knAI_color][ene_id]
            if ene_agent:
                actions = ene_agent.execute_before_all_actors(self.map)
                move_dis_dict = {}
                move_dixing_dict = {}
                sorted_move_list = []
                sorted_move_list2 = []
                attack_type_dict = {}
                for move_tmp in actions[0]:
                    hex = self.map.get_spec_len_dir_pos(ene_agent.cur_hex, 1, move_tmp)
                    dis = self.map.get_dis_between_hex(hex, mid_point)
                    move_dis_dict[move_tmp] = dis
                    move_dixing_dict[move_tmp] = self.map.map_data[hex // 100][hex % 100][0]
                if move_dis_dict != {}:
                    sorted_move_list = sorted(move_dis_dict.items(), key=lambda item: item[1], reverse=False)
                    sorted_move_list2 = sorted(move_dixing_dict.items(), key=lambda item: item[1], reverse=True)
                    # if sorted_move_list2[0][1] == 1:
                    #     print()

                for attack_tmp in actions[1]:
                    attack_type_dict[self.agentid_type_dict[attack_tmp]] = attack_tmp

                if actions[1] != []:
                    if 0 in attack_type_dict.keys():
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[0]] + 6
                    elif 1 in attack_type_dict.keys():
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[1]] + 6
                    else:
                        actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[2]] + 6
                else:
                    if sorted_move_list != []:
                        if ene_agent.type != 0:
                            if ene_agent.can_attack == []:
                                if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai:
                                    actions_dict[ene_id] = sorted_move_list[0][0]
                                else:
                                    if self.map.get_dis_between_hex(mid_point,
                                                                    ene_agent.cur_hex) > map_dis_for_ai - 2:
                                        if sorted_move_list2[0][1] == 1.0:
                                            actions_dict[ene_id] = sorted_move_list2[0][0]
                                        else:
                                            actions_dict[ene_id] = np.random.choice(actions[0])
                                    else:
                                        if self.map.map_data[ene_agent.cur_hex // 100][ene_agent.cur_hex % 100][
                                            0] == 0.0:
                                            actions_dict[ene_id] = np.random.choice(actions[0])
                        else:
                            if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai:
                                actions_dict[ene_id] = sorted_move_list[0][0]
                            else:
                                if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai - 2:
                                    if sorted_move_list2[0][1] == 1.0:
                                        actions_dict[ene_id] = sorted_move_list2[0][0]
                                    else:
                                        actions_dict[ene_id] = np.random.choice(actions[0])
                                else:
                                    if self.map.map_data[ene_agent.cur_hex // 100][ene_agent.cur_hex % 100][
                                        0] == 0.0:
                                        actions_dict[ene_id] = np.random.choice(actions[0])
        return actions_dict

    def action_space_sample_AI2(self, knAI_color):
        if self.game_map_id == 0:
            if knAI_color == 0:
                peo_dudian = 710
                zhanche_dundian = 808
            else:
                peo_dudian = 713
                zhanche_dundian = 415
        elif self.game_map_id == 2:
            if knAI_color == 0:
                peo_dudian = 407
                zhanche_dundian = 808
            else:
                peo_dudian = 1220
                zhanche_dundian = 719
        elif self.game_map_id == 3:
            if knAI_color == 0:
                peo_dudian = 312
                zhanche_dundian = 714
            else:
                peo_dudian = 2224
                zhanche_dundian = 1822
        elif self.game_map_id == 4:
            if knAI_color == 0:
                peo_dudian = 915
                zhanche_dundian = 1614
            else:
                peo_dudian = 1823
                zhanche_dundian = 1422
        else:
            if knAI_color == 0:
                peo_dudian = 2535
                zhanche_dundian = 2936
            else:
                peo_dudian = 3144
                zhanche_dundian = 3543
        import numpy as np
        if self.game_map_id == 0:
            map_dis_for_ai = 6
        else:
            map_dis_for_ai = 7
        mid_point = (self.map.map_size[self.game_map_id][0] - 1) / 2 * 100 + (
                    self.map.map_size[self.game_map_id][1] - 1) / 2
        mid_point = int(mid_point)
        actions_dict = {}
        for ene_id in self.agents_alive_list[knAI_color]:

            ene_agent = self.agents_alive_list[knAI_color][ene_id]
            if ene_agent:
                ene_type = ene_agent.type
                if ene_type == 2:
                    mid_point = peo_dudian
                if ene_type == 1:
                    mid_point = zhanche_dundian
                actions = ene_agent.execute_before_all_actors(self.map)
                move_dis_dict = {}
                sorted_move_list = []
                attack_type_dict = {}
                for move_tmp in actions[0]:
                    hex = self.map.get_spec_len_dir_pos(ene_agent.cur_hex, 1, move_tmp)
                    dis = self.map.get_dis_between_hex(hex, mid_point)
                    move_dis_dict[move_tmp] = dis
                if move_dis_dict != {}:
                    sorted_move_list = sorted(move_dis_dict.items(), key=lambda item: item[1], reverse=False)

                for attack_tmp in actions[1]:
                    attack_type_dict[self.agentid_type_dict[attack_tmp]] = attack_tmp

                guide_list = [0 for i in range(self.n_agents)]
                if self.game_map_id != 0:
                    if self.agentid_type_dict[ene_id] == 1:
                        guide_list = self.get_avail_guide_shoot()
                if sum(guide_list) != 0:
                    for index in range(len(guide_list)):
                        if guide_list[index] == 1:
                            actions_dict[ene_id] = index + 9
                else:
                    if actions[1] != []:
                        if 0 in attack_type_dict.keys():
                            actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[0]] + 6
                        elif 1 in attack_type_dict.keys():
                            actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[1]] + 6
                        else:
                            actions_dict[ene_id] = self.agentid_index_dict[1 - knAI_color][attack_type_dict[2]] + 6
                    else:
                        if sorted_move_list != []:
                            if ene_type == 0:
                                if self.map.get_dis_between_hex(mid_point, ene_agent.cur_hex) > map_dis_for_ai:
                                    actions_dict[ene_id] = sorted_move_list[0][0]
                                else:
                                    actions_dict[ene_id] = np.random.choice(actions[0])
                            else:
                                if ene_agent.cur_hex != mid_point:
                                    if ene_agent.can_attack == []:
                                        actions_dict[ene_id] = sorted_move_list[0][0]

        return actions_dict

    def render(self, src):
        # todo gai
        bg_color = (200, 200, 200)

        def draw_Hexagonal(scr):
            a = 5
            for i in range(0, self.map.MAP_X):
                for j in range(0, self.map.MAP_Y):
                    if i % 2 == 0:
                        dingdian = [((2 * j + 1) * a, 2 * i * a), ((2 * j + 1) * a + a, 2 * i * a + a),
                                    ((2 * j + 1) * a + a, 2 * i * a + 2 * a), ((2 * j + 1) * a, 2 * i * a + 3 * a),
                                    (2 * j * a, (2 * i * a + 2 * a)), (2 * j * a, (2 * i * a + a))]
                    else:
                        dingdian = [((j + 1) * 2 * a, 2 * a * i), ((j + 1) * a * 2 + a, 2 * a * i + a),
                                    ((j + 1) * 2 * a + a, 2 * a * i + 2 * a), ((j + 1) * 2 * a, 2 * a * i + 3 * a),
                                    ((j + 1) * 2 * a - a, 2 * a * i + 2 * a), ((j + 1) * 2 * a - a, 2 * a * i + a)]
                    if self.map.map_data[i][j][0] != 1.0:
                        pygame.draw.polygon(scr, (0, 200, 100), dingdian, True)
                    else:
                        pygame.draw.polygon(scr, (0, 200, 50), dingdian, 0)

        def draw_op(src):
            for color in [0, 1]:
                for agent_id in self.agents_alive_list[color].keys():
                    if self.agents_alive_list[color][agent_id] != None:
                        self.agents_alive_list[color][agent_id].draw(src)

        src.fill(bg_color)
        draw_Hexagonal(src)
        draw_op(src)


    def step(self, actions):
        actions_int = [int(a) for a in actions]
        info = {"battle_won": False}
        aly_blood_0, ene_blood_0 = self.get_blood_score()
        actions_ene = self.action_space_sample_AI2(1-self.owner)
        for ene_id in actions_ene:
            action_ene = actions_ene[ene_id]
            avail_actions = self.get_avail_agent_actions_fact(ene_id)
            assert (
                    avail_actions[action_ene] == 1
            ), "Agent {} cannot perform action {}".format(ene_id, action_ene)
            agent_color = self.agentid_color_dict[ene_id]
            if self.agents_alive_list[agent_color][ene_id] != None:
                if action_ene <= 5:
                    target_hex = self.map.get_neighbors(
                        self.agents_alive_list[agent_color][ene_id].cur_hex)[action_ene]
                    self.agents_alive_list[agent_color][ene_id].actor_move(self.clock, target_hex)
                if action_ene >= 6 and action_ene <= 8:
                    self.agents_alive_list[agent_color][ene_id].actor_shoot(self.clock)
                    myl_id = self.index_agentid_dict[1 - agent_color][action_ene - 6]
                    self.attack(ene_id, myl_id)

                if action_ene >= 9 and action_ene <= 11:
                    self.agents_alive_list[agent_color][ene_id].actor_shoot(self.clock)

                    ene_id_0 = self.index_agentid_dict[1 - agent_color][action_ene - 9]
                    self.guide_attack(ene_id, ene_id_0)
                    print('ene guide shoot', self.clock )
                    for idx in self.agents_alive_list[1-self.owner].keys():
                        if self.agents_alive_list[1-self.owner][idx] != None:
                            if self.agents_alive_list[1-self.owner][idx].type == 2:
                                peo_guide = self.agents_alive_list[1-self.owner][idx]
                                if peo_guide == None:
                                    raise
                                else:
                                    self.agents_alive_list[1-self.owner][idx].actor_shoot(self.clock)
            else:
                raise

        for agent_id, action in enumerate(actions_int):

            agent_id = self.index_agentid_dict[self.owner][agent_id]
            avail_actions = self.get_avail_agent_actions_fact(agent_id)
            if avail_actions[action] == 0:
                continue
            agent_color = self.agentid_color_dict[agent_id]
            if self.agents_alive_list[agent_color][agent_id] != None:
                if action <= 5:
                    target_hex = self.map.get_neighbors(
                        self.agents_alive_list[agent_color][agent_id].cur_hex)[action]
                    self.agents_alive_list[agent_color][agent_id].actor_move(self.clock, target_hex)
                if action >= 6 and action <= 8:
                    self.agents_alive_list[agent_color][agent_id].actor_shoot(self.clock)
                    ene_id = self.index_agentid_dict[1-agent_color][action-6]
                    self.attack(agent_id, ene_id)
                if action >= 9 and action <= 11:
                    self.agents_alive_list[agent_color][agent_id].actor_shoot(self.clock)

                    ene_id = self.index_agentid_dict[1 - agent_color][action - 9]
                    self.guide_attack(agent_id, ene_id)
                    self.guide_num += 1
                    for idx in self.agents_alive_list[self.owner].keys():
                        if self.agents_alive_list[self.owner][idx] != None:
                            if self.agents_alive_list[self.owner][idx].type == 2:
                                peo_guide = self.agents_alive_list[self.owner][idx]
                                if peo_guide == None:
                                    raise
                                else:
                                    self.agents_alive_list[self.owner][idx].actor_shoot(self.clock)
            else:
                raise

        for color in [0, 1]:
            for agent_id in self.agents_alive_list[color].keys():
                if self.agents_alive_list[color][agent_id] != None:
                    self.agents_alive_list[color][agent_id].actor_move_time()
                    self.agents_alive_list[color][agent_id].actor_stop(self.clock)
                    self.agents_alive_list[color][agent_id].actor_shoot_cooling(self.clock)
        self.update_agents_alive_list()
        self.update_agent_see_attack()

        self.clock += 1
        if self.clock == 600:
            done = True
            _, win_mark_list = self.is_done()
            self.timeouts += 1
        else:
            done, win_mark_list = self.is_done()

        aly_die = 0
        ene_die = 0
        for i in [0,1]:
            for id in self.agents_alive_list[i].keys():
                if self.agents_alive_list[i][id] == None:
                    if i == self.owner:
                        aly_die += 1
                    else:
                        ene_die += 1
        info["dead_allies"] = aly_die
        info["dead_enemies"] = ene_die
        if win_mark_list[self.owner] == False and win_mark_list[1-self.owner] == True \
            and not self.win_counted:
            info["battle_won"] = True
            self.battles_won += 1
            self.win_counted = True

        aly_blood_1, ene_blood_1 = self.get_blood_score()
        reward = aly_blood_1 - aly_blood_0 + ene_blood_0 - ene_blood_1

        if done:
            print('yao reward is:', reward)
            self.battles_game += 1
            print('my_color：', self.owner, '  end_my_blood:', aly_blood_1, '  end_ene_blood:', ene_blood_1, 'self.guide_num ', self.guide_num )
            if win_mark_list[self.owner] == False and win_mark_list[1 - self.owner] == True:
                self.win_n += 1
                # print('win,num is :', self.win_n, '  win rate is:', self.win_n / self.battles_game)
            elif aly_blood_1 > ene_blood_1:
                self.win_n += 1
                # print('win,num is :', self.win_n, '  win rate is:', self.win_n / self.battles_game)
            else:
                self.win_l += 1
                # print('lose, you loser', self.win_l, '  lose rate is:', self.win_l / self.battles_game)

        return reward, done, info

    def get_blood_score(self):
        aly_blood = 0
        ene_blood = 0
        for agent_id in self.agents_alive_list[self.owner]:
            if self.agents_alive_list[self.owner][agent_id] != None:
                aly_blood += self.agents_alive_list[self.owner][agent_id].blood

        for agent_id in self.agents_alive_list[1-self.owner]:
            if self.agents_alive_list[1-self.owner][agent_id] != None:
                ene_blood += self.agents_alive_list[1-self.owner][agent_id].blood
        return aly_blood, ene_blood

    def attack(self, attack_agent_id, be_attack_agent_id):
        attack_blood = 0
        random_seed = np.random.uniform(1, 11, 1)
        if attack_agent_id in self.agents_alive_list[0].keys():
            color_attack = 0
        else:
            color_attack = 1
        if be_attack_agent_id in self.agents_alive_list[0].keys():
            color_be_attack = 0
        else:
            color_be_attack = 1
        if color_attack == color_be_attack:
            raise
        if self.agents_alive_list[color_be_attack][be_attack_agent_id] != None:
            if self.agents_alive_list[color_be_attack][be_attack_agent_id].type == 2:
                attack_pro = self.agents_alive_list[color_attack][attack_agent_id].attack_peo_pro
                attack_pro_num = attack_pro * 10
                if random_seed <= attack_pro_num:
                    attack_blood = self.agents_alive_list[color_attack][attack_agent_id].attack_peo
            else:
                attack_pro = self.agents_alive_list[color_attack][attack_agent_id].attack_car_pro
                attack_pro_num = attack_pro * 10
                if random_seed <= attack_pro_num:
                    attack_blood = self.agents_alive_list[color_attack][attack_agent_id].attack_car
            self.agents_alive_list[color_be_attack][be_attack_agent_id].blood -= attack_blood

    def guide_attack(self, attack_agent_id, be_attack_agent_id):
        attack_blood = 0
        random_seed = np.random.uniform(1, 11, 1)
        if attack_agent_id in self.agents_alive_list[0].keys():
            color_attack = 0
        else:
            color_attack = 1
        if be_attack_agent_id in self.agents_alive_list[0].keys():
            color_be_attack = 0
        else:
            color_be_attack = 1
        if color_attack == color_be_attack:
            raise
        if self.agents_alive_list[color_be_attack][be_attack_agent_id] != None:
            if self.agents_alive_list[color_be_attack][be_attack_agent_id].type == 2:
                raise
            if random_seed <= self.guide_attack_pro * 10:
                attack_blood = self.guide_attack_value
            self.agents_alive_list[color_be_attack][be_attack_agent_id].blood -= attack_blood

    def close_game(self):
        pass

    def restore_state(self):
        pass


    def get_state_size(self):
        return 85

    def get_obs_size(self):
        return 85

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_stats(self):

        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def close(self):
        pass

    def save_replay(self):
        pass









