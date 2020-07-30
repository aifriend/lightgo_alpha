import random

from Utils import Utils


class Connection:

    def __init__(self, player_num, logger):
        self.player_num = player_num
        self.log = logger

    def get_possible_connections(self, lh_states, orig):
        possible_connections = []
        for dest in lh_states:
            # Do not connect with self
            # Do not connect if we have not the key
            # Do not connect if it is already connected
            # Do not connect if we do not own destiny
            # Do not connect if intersects
            if (dest != orig and
                    lh_states[dest]["have_key"] and
                    list(orig) not in lh_states[dest]["connections"] and
                    lh_states[dest]["owner"] == self.player_num and
                    not Utils.has_lhs(orig, dest, lh_states) and
                    not Utils.has_connections(lh_states, orig, dest)):
                possible_connections.append(dest)
        return possible_connections

    def decide_connection(self, possible_connections, state, lh_states):
        my_pos = tuple(state["position"])
        for conn in possible_connections:
            if Utils.closes_tri(lh_states, my_pos, conn):
                self.log("CONNECT TRI: %s", str(conn))
                return conn

        dest_lh = self._decide_dest_lh_connection(state, lh_states)
        if dest_lh is not None:
            self.log("CONNECT CEL: %s", str(dest_lh))
            return dest_lh
        else:
            conn = random.choice(possible_connections)
            self.log("CONNECT RANDOM: %s", str(conn))
            return conn

    def _decide_dest_lh_connection(self, state, lh_states):
        tri = None
        best_tri = 0
        my_pos = tuple(state["position"])
        possible_connections = self.get_possible_connections(lh_states, my_pos)
        if len(possible_connections) > 1:
            for dest_conn in possible_connections:
                third_possible_connections = possible_connections.copy()
                third_possible_connections.remove(dest_conn)
                for third_conn in third_possible_connections:
                    new_tri = len(Utils.closes_tri_by(my_pos, dest_conn, third_conn, True))
                    if new_tri > best_tri:
                        best_tri = new_tri
                        tri = third_conn

        return tri
