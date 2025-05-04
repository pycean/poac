
import numpy as np
import h5py
from src.envs.bq_src.map.m_exception import print_exception

class Map:

    def __init__(self, map_id):
        self.list_neighdir_offset_ji = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)]
        self.list_neighdir_offset_ou = [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
        self.map_size = {0: (13, 23), 2: (17, 27), 3: (27, 37), 4: (27, 37), 5: (67, 77)}
        self.terrain_id = map_id
        self.MAP_X = self.map_size[self.terrain_id][0]
        self.MAP_Y = self.map_size[self.terrain_id][1]
        self.map_data = h5py.File('src/envs/bq_src/map/'+'map_'+str(self.terrain_id)+'.h5','r')['map_data'][:]
        print()

    def is_pos_valid(self, pos):
        if 0 <= pos//100 < self.MAP_X and 0 <= pos % 100 < self.MAP_Y:
            return True
        return False

    def get_map_data(self):
        return self.map_data, self.MAP_X, self.MAP_Y

    def get_neighbors(self, pos):
        try:
            row, col = pos // 100, pos % 100
            assert self.is_pos_valid(pos)
            list_neigh_loc = []
            ji_flag = row % 2 == 1
            list_neighdir_offset = self.list_neighdir_offset_ji if ji_flag else self.list_neighdir_offset_ou
            for dir_index in range(6):
                list_neigh_loc.append(tuple(np.add((row, col), list_neighdir_offset[dir_index])))
            nei = []
            for e in list_neigh_loc:
                if self.is_pos_valid(e[0]*100 + e[1]):
                    nei.append(e[0]*100 + e[1])
                else:
                    nei.append(-1)
            return nei
        except Exception as e:
            raise e

    def get_dis_between_hex(self, pos1, pos2):
        try:
            row1 = pos1//100
            col1 = pos1%100
            row2 = pos2 // 100
            col2 = pos2 % 100
            q1 = col1 - (row1 - (row1 & 1)) // 2
            r1 = row1
            s1 = 0 - q1 - r1
            q2 = col2 - (row2 - (row2 & 1)) // 2
            r2 = row2
            s2 = 0 - q2 - r2
            return (abs(q1-q2) + abs(r1-r2) + abs(s1-s2))//2

        except Exception as e:
            print_exception(e)
            raise e

    def is_neighbor(self, pos1, pos2):

        return self.get_dis_between_hex(pos1=pos1, pos2=pos2) == 1

    def get_spec_len_dir_pos(self, pos, len, dir):
        try:
            assert self.is_pos_valid(pos)
            assert (dir >= 0 and dir < 6)
            row, col = pos // 100, pos % 100
            tt = (row, col)
            while len > 0:
                ji_flag = tt[0] % 2 == 1
                list_neighdir_offset = self.list_neighdir_offset_ji if ji_flag else self.list_neighdir_offset_ou
                nt = tuple(np.add((tt[0], tt[1]), list_neighdir_offset[dir]))
                #print(nt[0]*100+nt[1])
                if self.is_pos_valid(nt[0]*100 + nt[1]):
                    tt = nt
                else:
                    break
                len -= 1
            return tt[0]*100 + tt[1]
        except Exception as e:
            print_exception(e)
            raise e
