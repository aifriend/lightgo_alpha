import math

import numpy as np

from examples.lh import colinear, intersect, render, distt
from examples.lh.config.configuration import Configuration
from examples.lh.config.gameconfig import GameConfig, MoveError
from examples.lh.logic.board.island import Island
from examples.lh.logic.board.lighthouse import Lighthouse
from examples.lh.logic.board.player import Player


class Connection:
    def __init__(self):
        self._conns = set()

    @property
    def conns(self):
        return self._conns

    @staticmethod
    def available_board_conns(pieces):
        lh_connection_layer = pieces[:, :, Configuration.LH_CONN_IDX]
        row, col = lh_connection_layer.shape
        for y in range(row):
            cx0, cy0, cx1, cy1 = lh_connection_layer[y, :4]
            if cx0:
                yield frozenset(((cx0, cy0), (cx1, cy1)))

    def board_to_conns(self, pieces):
        for conn in Connection.available_board_conns(pieces):
            (cx0, cy0), (cx1, cy1) = conn
            self._conns.add(frozenset(((int(cx0), int(cy0)), (int(cx1), int(cy1)))))

    def conns_to_board(self, pieces):
        idx = 0
        lh_connection_layer = pieces[:, :, Configuration.LH_CONN_IDX]
        for (cx0, cy0), (cx1, cy1) in self._conns:
            lh_connection_layer[idx, Configuration.CONN_X0_IDX] = cx0
            lh_connection_layer[idx, Configuration.CONN_Y0_IDX] = cy0
            lh_connection_layer[idx, Configuration.CONN_X1_IDX] = cx1
            lh_connection_layer[idx, Configuration.CONN_Y1_IDX] = cy1
            idx += 1

    @staticmethod
    def encode_conns(pieces, player_id) -> None:
        n_conns = 0
        player_lh_owned_pose = Lighthouse.owned_by(pieces, player=player_id)
        for conn in Connection.available_board_conns(pieces):
            if next(iter(conn)) in player_lh_owned_pose:
                n_conns += 1
        pieces[:, :, Configuration.LH_CONN_IDX] = 0
        pieces[0][0][Configuration.LH_CONN_IDX] = n_conns

    def close_conn(self, pos):
        self._conns = set(i for i in self._conns if pos not in i)

    @staticmethod
    def clear_board(pieces):
        pieces[:, :, Configuration.LH_CONN_IDX] = 0


class Polygon:
    def __init__(self):
        self._tris = dict()

    @property
    def tris(self):
        return self._tris

    @staticmethod
    def available_board_tris(pieces):
        lh_polygon_layer = pieces[:, :, Configuration.LH_TRI_IDX]
        row, col = lh_polygon_layer.shape
        for y in range(row):
            tx0, ty0, tx1, ty1, tx2, ty2, tc = lh_polygon_layer[y, :7]
            if tx0:
                yield ((tx0, ty0), (tx1, ty1), (tx2, ty2)), tc

    def board_to_tris(self, pieces):
        for key, c_tris in Polygon.available_board_tris(pieces):
            (tx0, ty0), (tx1, ty1), (tx2, ty2) = key
            idx_key = (int(tx0), int(ty0)), (int(tx1), int(ty1)), (int(tx2), int(ty2))
            self._tris[idx_key] = int(c_tris)

    def tris_to_board(self, pieces):
        idx = 0
        lh_polygon_layer = pieces[:, :, Configuration.LH_TRI_IDX]
        for key, cells in self.tris.items():
            (tx0, ty0), (tx1, ty1), (tx2, ty2) = key
            lh_polygon_layer[idx, Configuration.TRI_X0_IDX] = tx0
            lh_polygon_layer[idx, Configuration.TRI_Y0_IDX] = ty0
            lh_polygon_layer[idx, Configuration.TRI_X1_IDX] = tx1
            lh_polygon_layer[idx, Configuration.TRI_Y1_IDX] = ty1
            lh_polygon_layer[idx, Configuration.TRI_X2_IDX] = tx2
            lh_polygon_layer[idx, Configuration.TRI_Y2_IDX] = ty2
            lh_polygon_layer[idx, Configuration.TRI_CC_IDX] = cells
            idx += 1

    @staticmethod
    def encode_tris(pieces, player_id) -> None:
        n_cell = 0
        player_lh_owned_pose = Lighthouse.owned_by(pieces, player=player_id)
        for key, ctris in Polygon.available_board_tris(pieces):
            if next(iter(key)) in player_lh_owned_pose:
                n_cell += ctris
        pieces[:, :, Configuration.LH_TRI_IDX] = 0
        pieces[0][0][Configuration.LH_TRI_IDX] = n_cell

    def close_tri(self, pos):
        self._tris = dict(i for i in self._tris.items() if pos not in i[0])

    @staticmethod
    def clear_board(pieces):
        pieces[:, :, Configuration.LH_TRI_IDX] = 0


class Board(object):
    RDIST = 5

    # All possible movements: (row, col)
    POSSIBLE_MOVES = {0: (0, 0),  # pass
                      1: (0, -1),  # down
                      2: (0, 1),  # up
                      3: (1, 0),  # right
                      4: (-1, 0),  # left
                      5: (1, -1),  # upright
                      6: (-1, -1),  # upleft
                      7: (1, 1),  # downright
                      8: (-1, 1)}  # downleft

    # All possible attacks
    POSSIBLE_ATTACK = {9: 0.1,  # attack 10%
                       10: 0.3,  # attack 30%
                       11: 0.6,  # attack 60%
                       12: 0.8,  # attack 80%
                       13: 1.0}  # attack 100%

    # All possible connections
    POSSIBLE_CONNECTIONS = {18: "connect4",
                            17: "connect3",
                            16: "connect2",
                            15: "connect1",
                            14: "connect0"}

    def __init__(self):
        self._size = tuple()
        self._island = None
        self._players = dict()
        self._lighthouses = dict()
        self._connection = Connection()
        self._polygon = Polygon()

    @property
    def size(self):
        return self._size

    @property
    def island(self):
        return self._island

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, value):
        self._players = value

    @property
    def lighthouses(self):
        return self._lighthouses

    @lighthouses.setter
    def lighthouses(self, value):
        self._lighthouses = value

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, value):
        self._connection = value

    @property
    def poligon(self):
        return self._polygon

    @poligon.setter
    def poligon(self, value):
        self._polygon = value

    """
    INIT & CONVERTERS
    """

    def init(self, cfg_file):
        # board from file
        cfg = GameConfig(cfg_file)
        self._size = tuple(cfg.size)

        # island
        self._island = Island.init(cfg.f_island)

        # players
        self._players = Player.init(cfg.f_players)

        # lighthouses
        self._lighthouses = Lighthouse.init(self, cfg.f_lighthouses)

        # connections
        self._connection = Connection()
        self._polygon = Polygon()

    def from_array(self, pieces):
        row, col, enc = pieces.shape
        self._size = (row, col)

        # island
        self._island = Island.from_array(pieces)

        # players
        self._players = Player.from_array(pieces)

        # lighthouses
        self._lighthouses = Lighthouse.from_array(pieces, self)

        # conns & tris
        self._connection.board_to_conns(pieces)
        self._polygon.board_to_tris(pieces)

    def to_array(self, pieces) -> np.array:
        new_pieces = np.copy(pieces)

        # island
        new_pieces[:, :, Configuration.ISLAND_IDX] = self._island.island_map

        # energy
        new_pieces[:, :, Configuration.ENERGY_IDX] = self._island.energy.energy_map

        # reset board
        Player.clear_board(new_pieces)
        Lighthouse.clear_board(new_pieces)
        Connection.clear_board(new_pieces)
        Polygon.clear_board(new_pieces)

        # players
        for player in self._players.values():
            player.pl_turn_to_board(new_pieces)
            player.pl_energy_to_board(new_pieces)
            player.pl_keys_to_board(new_pieces)
            player.pl_score_to_board(new_pieces)

        # lighthouses
        for lighthouse in self._lighthouses.values():
            lighthouse.lh_to_board(new_pieces)
            lighthouse.lh_energy_to_board(new_pieces)
            lighthouse.lh_owner_to_board(new_pieces)

        # conns & tris
        self._connection.conns_to_board(new_pieces)
        self._polygon.tris_to_board(new_pieces)

        return np.array(new_pieces)

    def close_connection(self, pos):
        self._connection.close_conn(pos)
        self._polygon.close_tri(pos)

    """
    GAME UPDATES
    """

    def pre_round(self):
        # Update island energy
        for pos in self._lighthouses.keys():
            for y in range(pos[1] - self.RDIST + 1, pos[1] + self.RDIST):
                for x in range(pos[0] - self.RDIST + 1, pos[0] + self.RDIST):
                    dist = distt(pos, (x, y))
                    delta = int(math.floor((self.RDIST - dist) / 2))
                    if delta > 0:
                        self._island.energy[x, y] += delta

        # Decay lighthouse energy
        for lh in self._lighthouses.values():
            decay = int(Lighthouse.DECAY / 2)
            lh.decay(decay)

    def pre_player_update(self, player):
        # get player
        r_player = self.player_by(player)
        if not r_player:
            return

        # Player get lh keys
        player_posmap = dict()
        for player in self._players.values():
            if player.pos in player_posmap:
                player_posmap[player.pos].append(player)
            else:
                player_posmap[player.pos] = [player]
            if player.pos == r_player.pos:
                if player.pos in self._lighthouses:
                    player.keys.add(player.pos)

        # Update player energy
        for pos, players in player_posmap.items():
            if pos == r_player.pos:
                energy = self._island.energy[pos] // len(players)
                for player in players:
                    player.energy += energy
                self._island.energy[pos] = 0

    def post_player_update(self, player):
        # get player
        r_player = self.player_by(player)
        if not r_player:
            return

        # Evaluate board player score: lighthouses owned
        for lh in self._lighthouses.values():
            if lh.owner is not None:
                if lh.owner == r_player.turn:
                    r_player.score += 2

        # Evaluate board player score: lighthouses linked
        for pair in self._connection.conns:
            lh = self._lighthouses[next(iter(pair))]
            if lh.owner == r_player.turn:
                r_player.score += 2

        # Evaluate board player score: lighthouses closed
        for tri, cells in self._polygon.tris.items():
            lh = self._lighthouses[tri[0]]
            if lh.owner == r_player.turn:
                r_player.score += cells

    def connect(self, player, dest_pos):
        if player.pos not in self._lighthouses.keys():
            raise MoveError("Player must be located at the origin lighthouse")
        if dest_pos not in self._lighthouses.keys():
            raise MoveError("Destination must be an existing lighthouse")
        orig = self._lighthouses[player.pos]
        dest = self._lighthouses[dest_pos]
        if orig.owner != player.turn or dest.owner != player.turn:
            raise MoveError("Both lighthouses must be player-owned")
        if dest.pos not in player.keys:
            raise MoveError("Player does not have the destination key")
        if orig is dest:
            raise MoveError("Cannot connect lighthouse to itself")
        assert orig.energy and dest.energy
        pair = frozenset((orig.pos, dest.pos))
        if pair in self._connection.conns:
            raise MoveError("Connection already exists")
        x0, x1 = sorted((orig.pos[0], dest.pos[0]))
        y0, y1 = sorted((orig.pos[1], dest.pos[1]))
        for lh in self._lighthouses.values():
            if (x0 <= lh.pos[0] <= x1 and y0 <= lh.pos[1] <= y1 and
                    lh.pos not in (orig.pos, dest.pos) and
                    colinear(orig.pos, dest.pos, lh.pos)):
                raise MoveError("Connection cannot intersect a lighthouse")
        new_tris = set()
        for c in self._connection.conns:
            if intersect(tuple(c), (orig.pos, dest.pos)):
                raise MoveError("Connection cannot intersect another connection")
            if orig.pos in c:
                third = next(t for t in c if t != orig.pos)
                if frozenset((third, dest.pos)) in self._connection.conns:
                    new_tris.add((orig.pos, dest.pos, third))

        player.keys.remove(dest.pos)
        assert not isinstance(orig.pos[0], float) or not isinstance(dest.pos[0], float)
        self._connection.conns.add(pair)
        for i in new_tris:
            self._polygon.tris[i] = len([j for j in render(i) if self._island[j]])

    """
    MOVES DETECTIONS
    """

    # Direction moves
    def available_worker_moves(self, player) -> (int, (int, int)):
        # Iter possible movements
        cx, cy = player.pos
        ox, oy = self.player_by(-player.turn).pos
        for key, mov in Board.POSSIBLE_MOVES.items():
            if self._island[cx + mov[0], cy + mov[1]] and (ox, oy) != (cx + mov[0], cy + mov[1]):
                yield (key, mov)
        yield (None, None)

    def get_available_worker_actions(self, player, move):
        # Get possible worker actions
        for (key, action) in self.available_worker_moves(player):
            if key and key == move:
                return action
        return None

    def get_available_worker_moves(self, player, valid_moves):
        # Get possible worker movements
        for (key, _) in self.available_worker_moves(player):
            if key:
                valid_moves[key] = 1
        return valid_moves

    # Attack moves
    def available_attack_moves(self, player) -> (int, (int, int)):
        # Check possible attacks
        if player.pos in self._lighthouses.keys():
            if player.energy > 10:  # min energy valid for attack
                # max energy available selected for attack
                attack_energy = int(round(player.energy * Board.POSSIBLE_ATTACK[13]))
                if attack_energy > Lighthouse.DECAY:
                    yield (13, attack_energy)
                attack_energy = int(round(player.energy * Board.POSSIBLE_ATTACK[12]))
                if attack_energy > Lighthouse.DECAY:
                    yield (12, attack_energy)
                attack_energy = int(round(player.energy * Board.POSSIBLE_ATTACK[11]))
                if attack_energy > Lighthouse.DECAY:
                    yield (11, attack_energy)
                attack_energy = int(round(player.energy * Board.POSSIBLE_ATTACK[10]))
                if attack_energy > Lighthouse.DECAY:
                    yield (10, attack_energy)
                attack_energy = int(round(player.energy * Board.POSSIBLE_ATTACK[9]))
                if attack_energy > Lighthouse.DECAY:
                    yield (9, attack_energy)
        yield (None, None)

    def get_available_attack_actions(self, player, move):
        # Get possible attack actions
        for key, value in self.available_attack_moves(player):
            if key and key == move:
                return value
        return None

    def get_available_attack_moves(self, player, valid_moves):
        # Get possible attack movements
        for key, _ in self.available_attack_moves(player):
            if key:
                valid_moves[key] = 1
        return valid_moves

    # Connection moves
    def available_lh_connection_moves(self, player) -> (int, (int, int)):
        # Check possible connections
        possible_connections = Board.POSSIBLE_CONNECTIONS.copy()
        if player.pos in self._lighthouses.keys():
            orig = self._lighthouses[player.pos]
            if orig.owner == player.turn:  # own lighthouse
                for dest in self._lighthouses.values():
                    # Do not connect with self
                    # Do not connect if we have not the key
                    if dest.pos != orig.pos and player.has_key(dest.pos):
                        dest_connections = list()
                        for c in self._connection.conns:
                            if dest.pos in c:
                                for pair in c:
                                    if dest.pos != pair:
                                        dest_connections.append(pair)
                        # Do not connect if it is already connected
                        # Do not connect if we do not own destiny
                        # Do not connect if intersects
                        not_conn = orig.pos not in dest_connections
                        owned = dest.owner == player.turn
                        not_colinear = not self._lh_colinear(orig.pos, dest.pos)
                        not_intersect = not self._lh_intersect(orig.pos, dest.pos)
                        if not_conn and owned and not_colinear and not_intersect:
                            try:
                                key, _ = possible_connections.popitem()
                                yield (key, dest)
                            except KeyError:
                                break
        yield (None, None)

    def get_available_lh_connection_actions(self, player, move):
        # Get possible connection actions
        for (key, dest) in self.available_lh_connection_moves(player):
            if key and key == move:
                return dest.pos
        assert False

    def get_available_lh_connection_moves(self, player, valid_moves):
        # Get possible connection movements
        for (key, _) in self.available_lh_connection_moves(player):
            if key:
                valid_moves[key] = 1
        return valid_moves

    def player_by(self, turn):
        for player in self._players.values():
            if player.turn == turn:
                return player
        assert turn == 0 or turn == -1 or turn == 1

    """
    TOOLS
    """

    def _lh_colinear(self, orig, dest):
        x0, x1 = sorted((orig[0], dest[0]))
        y0, y1 = sorted((orig[1], dest[1]))
        for lh in self._lighthouses.values():
            if (x0 <= lh.pos[0] <= x1 and y0 <= lh.pos[1] <= y1 and
                    lh.pos not in (orig, dest) and
                    self._colinear(orig, dest, lh.pos)):
                return True

        return False

    def _lh_intersect(self, orig, dest):
        for lh in self._lighthouses.values():
            connections = [next(l for l in c if l is not lh.pos)
                           for c in self._connection.conns if lh.pos in c]
            for c in connections:
                if self._intersect((lh.pos, tuple(c)), (orig, dest)):
                    return True

        return False

    def _colinear(self, a, b, c):
        return self._orient2d(a, b, c) == 0

    @staticmethod
    def _orient2d(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

    def _intersect(self, j, k):
        j1, j2 = j
        k1, k2 = k
        return (
                self._orient2d(k1, k2, j1) *
                self._orient2d(k1, k2, j2) < 0 and
                self._orient2d(j1, j2, k1) *
                self._orient2d(j1, j2, k2) < 0
        )
