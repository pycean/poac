class Op_Env:
    def __init__(self, color, init_hex, id, type):
        self.color = color
        self.id = id
        self.type = type

        self.cur_hex = init_hex

    def execute_before_all_actors(self, map):
        can_move = []
        can_attack = []
        if self.move_time == 0:
            if map.is_pos_valid(self.cur_hex) == False:
                raise
            neibors = map.get_neighbors(self.cur_hex)
            for i in range(len(neibors)):
                if neibors[i] != -1:
                    can_move.append(i)
        else:
            can_move = []

        if self.can_attack != []:
            if self.shoot_cooling_time >= self.shoot_cooling_time_cost \
                    and self.stop_time >= self.stop_can_shoot_time:
                can_attack = self.can_attack
            else:
                can_attack = []

        return [can_move, can_attack]


    def actor_move(self, clock, target_hex=-1):
        if self.move_time == 0:
            self.move_time = self.cost_speed
            self.move_start_clock = clock
            self.move_target_hex = target_hex

    def actor_move_time(self):
        if self.move_time > 0:
            self.move_time -= 1
            if self.move_time == 0:
                self.cur_hex = self.move_target_hex


    def actor_stop(self,clock):
        if self.move_time == 0 and self.move_start_clock != clock:
            # 以防1s减下去，执行和停止同步进行，其他的情况下只要move_time == 0都是可以stop time ++
            self.stop_time += 1
        if self.move_time > 0:
            self.stop_time = 0
        if self.move_time == 0 and self.move_start_clock == clock:
            self.stop_time = 0


    def actor_shoot(self, clock):

        self.short_start_clock = clock

    def actor_shoot_cooling(self, clock):

        if self.short_start_clock == clock:
            self.shoot_cooling_time = 0
        else:
            self.shoot_cooling_time += 1

