from Utils import Utils


class LightHouse:
    def __init__(self, lighthouses, init_state):
        if init_state is None:
            self.lh_dist_maps = list()
        else:
            self.lh_dist_maps = {
                lh: self._get_lh_dist_map(lh, init_state["map"])
                for lh in lighthouses
            }

    @staticmethod
    def _get_lh_dist_map(lh, world_map):
        lh_max_dist = max(len(world_map), len(world_map[0])) + 1
        lh_map = [[-1 if pos else lh_max_dist for pos in row] for row in world_map]
        lh_map[lh[1]][lh[0]] = 0
        dist = 1
        points = Utils.get_possible_points(lh, lh_map)
        while len(points):
            next_points = []
            for x, y in points:
                lh_map[y][x] = dist
            for x, y in points:
                cur_points = Utils.get_possible_points((x, y), lh_map)
                next_points.extend(cur_points)
            points = list(set(next_points))
            dist += 1

        lh_map = [[lh_max_dist if pos == -1 else pos
                   for pos in row]
                  for row in lh_map]

        return lh_map

    def get_lh_states(self, state):
        my_pos = tuple(state["position"])
        dists_to_lhs = {
            lh: self.lh_dist_maps[lh][my_pos[1]][my_pos[0]]
            for lh in self.lh_dist_maps
        }

        _lh_states = {
            tuple(lh["position"]): lh
            for lh in state["lighthouses"]
        }

        for lh in _lh_states:
            _lh_states[lh]['cur_dist'] = \
                dists_to_lhs[tuple(_lh_states[lh]["position"])]

        return _lh_states

    def get_closest_lh(self, player_pose: tuple, lh_pose_list: list):
        dists_to_lhs = {
            lh: self.lh_dist_maps[lh][player_pose[1]][player_pose[0]]
            for lh in self.lh_dist_maps
        }
        lh_filter = filter(
            lambda y: y[0] not in lh_pose_list, dists_to_lhs.items())
        lh_closest = min(lh_filter, key=lambda x: x[1])

        return lh_closest

    def get_lh_dist(self, player_pose: tuple, lh_pose):
        dists_to_lhs = {
            lh: self.lh_dist_maps[lh][player_pose[1]][player_pose[0]]
            for lh in self.lh_dist_maps
        }
        lh_closest_dist = min(filter(
            lambda y: y[0] == lh_pose, dists_to_lhs.items()),
            key=lambda x: x[1])

        return lh_closest_dist
