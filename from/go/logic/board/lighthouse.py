from examples.lh.config.configuration import Configuration
from examples.lh.config.gameconfig import MoveError


class Lighthouse(object):
    DECAY = 10

    def __init__(self, game, pos):
        self._game = game
        self._pos = pos
        self._owner = 0
        self._energy = 0

    @property
    def pos(self):
        return self._pos

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        self._owner = value

    @property
    def energy(self):
        return self._energy

    def lh_to_board(self, pieces):
        pieces[self._pos[1], self._pos[0], Configuration.A_TYPE_IDX] = Configuration.d_a_type["Lighthouse"]

    def lh_energy_to_board(self, pieces, energy=None, add_energy=None):
        if energy and energy >= 0:
            self._energy = energy
        if add_energy and add_energy >= 0:
            self._energy += add_energy
        pieces[self._pos[1], self._pos[0], Configuration.LH_ENERGY_IDX] = int(round(self._energy))

    def lh_owner_to_board(self, pieces, owner=None):
        if owner is not None:
            self._owner = owner
        pieces[self._pos[1], self._pos[0], Configuration.LH_OWNER_IDX] = int(round(self._owner))

    def board_to_lh_energy(self, pieces):
        self._energy = int(round(pieces[self._pos[1], self._pos[0], Configuration.LH_ENERGY_IDX]))

    def board_to_lh_owner(self, pieces):
        self._owner = int(round(pieces[self._pos[1], self._pos[0], Configuration.LH_OWNER_IDX]))

    @staticmethod
    def clear_board(pieces):
        pieces[:, :, Configuration.LH_ENERGY_IDX] = 0
        pieces[:, :, Configuration.LH_OWNER_IDX] = 0

    @staticmethod
    def init(board, lighthouses):
        return dict((pos, Lighthouse(board, pos)) for i, pos in enumerate(lighthouses))

    @staticmethod
    def from_array(pieces, game):
        lh_list = dict()
        row, col, enc = pieces.shape
        for y in range(row):
            for x in range(col):
                lh_pose = (x, y)
                a_type = pieces[y, x, Configuration.A_TYPE_IDX]
                if a_type == Configuration.d_a_type["Lighthouse"]:
                    lighthouse = Lighthouse(game, lh_pose)
                    lighthouse.board_to_lh_energy(pieces)
                    lighthouse.board_to_lh_owner(pieces)
                    lh_list[lh_pose] = lighthouse
        return lh_list

    def attack(self, player, strength):
        if not isinstance(strength, int):
            raise MoveError("Strength must be an int")
        if strength < 0:
            raise MoveError("Strength must be positive")
        if strength > player.energy:
            strength = player.energy
        player.energy -= strength
        if self._owner is not None and self._owner != player.turn:
            d = min(self._energy, strength)
            self.decay(d)
            strength -= d
        if strength:
            self._owner = player.turn
            self._energy = strength

    def decay(self, by):
        self._energy -= by
        if self._energy <= 0:
            self._energy = 0
            self._owner = 0
            self._game.close_connection(self._pos)

    @staticmethod
    def owned_by(pieces, player=1):
        """
        Lighthouses owned by given player

        :param pieces:
        :param player:
        :return:
        """
        lh_owned = list()
        row, col, enc = pieces.shape
        for y in range(row):
            for x in range(col):
                if pieces[y, x, Configuration.LH_OWNER_IDX] == player:
                    lh_owned.append((y, x))

        return lh_owned
