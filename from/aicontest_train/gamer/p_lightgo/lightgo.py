from Connection import Connection
from Interface import Bot, Interface
from LightHouse import LightHouse
from Motion import Motion


class LightGo(Bot):
    NAME = "Lightgo"
    MAX_INT = 1e40

    def __init__(self, init_state=None):
        super().__init__(init_state)
        self.lh_service = LightHouse(self.lighthouses, init_state)
        self.connect_service = Connection(self.player_num, self.log)
        self.motion_service = Motion(self.map, self.player_num, self.log)

    def initialize(self, init_state):
        super().__init__(init_state)
        self.lh_service = LightHouse(self.lighthouses, init_state)
        self.connect_service = Connection(self.player_num, self.log)
        self.motion_service = Motion(self.map, self.player_num, self.log)

    def play(self, state):
        lh_states = self.lh_service.get_lh_states(state)
        my_pos = tuple(state["position"])

        if my_pos in lh_states:
            # Connect
            if lh_states[my_pos]["owner"] == self.player_num:
                possible_connections = \
                    self.connect_service.get_possible_connections(lh_states, my_pos)
                if possible_connections:
                    conn = self.connect_service.decide_connection(
                        possible_connections, state, lh_states)
                    return {
                        "command": "connect",
                        "destination": conn
                    }

            # Attack
            if 10 < state["energy"] >= lh_states[my_pos]["energy"]:  # 100
                energy = state["energy"]
                self.log("ATTACK TO: %s", str(my_pos))
                return {
                    "command": "attack",
                    "energy": energy
                }

        # Move
        move = self.motion_service.decide_movement(
            state, self.lh_service.lh_dist_maps, lh_states, self.connect_service)
        if move:
            return {
                "command": "move",
                "x": move[0],
                "y": move[1]
            }
        else:
            # Pass
            return {
                "command": "pass",
            }


if __name__ == "__main__":
    i_face = Interface(LightGo)
    i_face.run()
