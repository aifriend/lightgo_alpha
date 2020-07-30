import random

from Utils import Utils


class Motion:

    def __init__(self, game_map, player_num, logger):
        self.map = game_map
        self.player_num = player_num
        self.log = logger

    def decide_movement(self, state, lh_service, lh_states, con_service):
        possible_moves = self.get_possible_moves(state["position"])
        if state["energy"] < 500:
            move, energy_gain = Utils.harvest_movement(state["view"], possible_moves)
            if energy_gain > 10:
                self.log("MOVE TO HARVEST: %s", str(move))
                return move
        dest_lh = self._decide_dest_lh_movement(state, lh_states, con_service)
        move = self._to_lh_movement(dest_lh,
                                    lh_service,
                                    state["position"],
                                    possible_moves)

        self.log("MOVE TO LH: %s", str(move))
        return move

    def _decide_dest_lh_movement(self, state, lh_states, con_service):
        # Go to a interesting lighthouse
        for dest_lh in lh_states:
            lh_points = random.uniform(0.0, 1.0)
            lh_points -= lh_states[dest_lh]['cur_dist']
            if lh_states[dest_lh]["owner"] == self.player_num:
                if not lh_states[dest_lh]["have_key"]:
                    lh_points += 100
                if lh_states[dest_lh]["energy"] < state["energy"]:
                    lh_points += 50
            else:
                possible_connections = \
                    con_service.get_possible_connections(lh_states, dest_lh)
                lh_points += len(possible_connections) * 200
                if len(possible_connections) > 1:
                    for orig_conn in possible_connections:
                        for dest_conn in lh_states[orig_conn]["connections"]:
                            if tuple(dest_conn) in possible_connections:
                                tri_size = Utils.closes_tri(lh_states, dest_conn, orig_conn, size=True)
                                lh_points += 1000 * tri_size

                if lh_states[dest_lh]["energy"] < state["energy"]:
                    lh_points += 40
            lh_states[dest_lh]['points'] = lh_points

        dest_lh = max(lh_states.items(),
                      key=lambda x: x[1]['points'])[0]
        return dest_lh

    @staticmethod
    def _to_lh_movement(lh, lh_service, my_pos, possible_moves):
        dist_map = lh_service[lh]
        dist = {
            move: dist_map[move[1] + my_pos[1]][move[0] + my_pos[0]]
            for move in possible_moves
        }
        move = min(dist, key=dist.get)

        return move

    def get_possible_moves(self, pos):
        # Random move
        moves = ((-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1))

        # Check possible movements
        cx, cy = pos

        moves = sorted([(x, y) for x, y in moves if self.map[cy + y][cx + x]])
        return moves

    def is_in_island(self, pose):
        x, y = pose
        return self.map[y][x]
