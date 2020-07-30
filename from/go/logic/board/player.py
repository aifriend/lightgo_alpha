from examples.lh.config.configuration import Configuration
from examples.lh.config.gameconfig import MoveError


class Player(object):
    def __init__(self, pose, turn=0):
        self._turn = turn
        self._pos = pose
        self._score = 0
        self._energy = 0
        self._keys = set()

    @property
    def turn(self):
        return self._turn

    @turn.setter
    def turn(self, value):
        self._turn = value

    @property
    def pos(self):
        return self._pos

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value

    def has_key(self, lh_pose):
        return lh_pose in self._keys

    @staticmethod
    def clear_board(pieces):
        pieces[:, :, Configuration.P_NAME_IDX] = 0
        pieces[:, :, Configuration.PL_SCORE_W1_IDX] = 0
        pieces[:, :, Configuration.PL_SCORE_W2_IDX] = 0
        pieces[:, :, Configuration.PL_ENERGY_W1_IDX] = 0
        pieces[:, :, Configuration.PL_ENERGY_W2_IDX] = 0
        pieces[:, :, Configuration.LH_KEY_IDX] = 0

    @staticmethod
    def init(players):
        player_list = dict()
        for i, (c, pos) in enumerate(players[:2]):
            player_list[pos] = Player(pos, (-1 if i == 0 else 1))
        return player_list

    @staticmethod
    def from_array(pieces):
        player_list = dict()
        row, col, enc = pieces.shape
        for y in range(row):
            for x in range(col):
                p_turn = pieces[y, x, Configuration.P_NAME_IDX]
                if p_turn != 0:
                    player = Player((x, y), p_turn)
                    player.board_to_pl_turn(pieces)
                    player.board_to_pl_energy(pieces)
                    player.board_to_pl_score(pieces)
                    player.board_to_pl_keys(pieces)
                    player_list[(x, y)] = player
        return player_list

    def pl_turn_to_board(self, pieces, turn=None):
        if turn is not None:
            self._turn = turn
        pieces[self._pos[1], self._pos[0], Configuration.P_NAME_IDX] = int(round(self._turn))

    def pl_score_to_board(self, pieces, score=None):
        if score and score >= 0:
            self._score = score
        if self._turn == 1:
            pieces[self._pos[1], self._pos[0], Configuration.PL_SCORE_W1_IDX] = int(round(self._score))
        elif self._turn == -1:
            pieces[self._pos[1], self._pos[0], Configuration.PL_SCORE_W2_IDX] = int(round(self._score))

    def pl_energy_to_board(self, pieces, energy=None):
        if energy and energy >= 0:
            self._energy = energy
        if self._turn == 1:
            pieces[self._pos[1], self._pos[0], Configuration.PL_ENERGY_W1_IDX] = int(round(self._energy))
        elif self._turn == -1:
            pieces[self._pos[1], self._pos[0], Configuration.PL_ENERGY_W2_IDX] = int(round(self._energy))

    def pl_keys_to_board(self, pieces):
        for key in self._keys:
            lh_actual_key = pieces[key[1], key[0], Configuration.LH_KEY_IDX]
            if (lh_actual_key == 1 and self._turn == -1) or (lh_actual_key == -1 and self._turn == 1):
                pieces[key[1], key[0], Configuration.LH_KEY_IDX] = 3
            else:
                pieces[key[1], key[0], Configuration.LH_KEY_IDX] = int(round(self._turn))

    def board_to_pl_turn(self, pieces):
        self._turn = int(round(pieces[self._pos[1], self._pos[0], Configuration.P_NAME_IDX]))

    def board_to_pl_score(self, pieces):
        if self._turn == 1:
            self._score = int(round(pieces[self._pos[1], self._pos[0], Configuration.PL_SCORE_W1_IDX]))
        else:
            self._score = int(round(pieces[self._pos[1], self._pos[0], Configuration.PL_SCORE_W2_IDX]))

    def board_to_pl_energy(self, pieces):
        if self._turn == 1:
            self._energy = int(round(pieces[self._pos[1], self._pos[0], Configuration.PL_ENERGY_W1_IDX]))
        else:
            self._energy = int(round(pieces[self._pos[1], self._pos[0], Configuration.PL_ENERGY_W2_IDX]))

    def board_to_pl_keys(self, pieces):
        row, col, enc = pieces.shape
        for y in range(row):
            for x in range(col):
                actor_type = pieces[y, x, Configuration.A_TYPE_IDX]
                if actor_type == Configuration.d_a_type["Lighthouse"]:
                    lh_key_owner = pieces[y, x, Configuration.LH_KEY_IDX]
                    if lh_key_owner == self._turn:
                        self._keys.add((x, y))

    def move(self, island, delta):
        dx, dy = delta
        if dx not in (0, 1, -1) or dy not in (0, 1, -1):
            raise MoveError("Delta must be 1 cell away")
        new_x, new_y = self._pos[0] + dx, self._pos[1] + dy
        if not island[new_y, new_x]:
            raise MoveError("Target pos is not in island")
        self._pos = (new_x, new_y)
