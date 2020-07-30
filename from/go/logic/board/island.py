import numpy as np

from examples.lh import distt
from examples.lh.config.configuration import Configuration


class Island(object):
    MAX_ENERGY = 100

    def __init__(self, island_map, energy_map=None):
        self._island_map = np.array(island_map)
        self.h = len(self._island_map)
        self.w = len(self._island_map[0])
        if energy_map is not None:
            self._energy_map = np.array(energy_map)
        else:
            self._energy_map = np.zeros((self.h, self.w))

        class _Energy(object):
            def __getitem__(unused, pos):
                xx, yy = pos
                if self[pos]:
                    return self._energy_map[yy][xx]
                else:
                    return 0

            def __setitem__(unused, pos, val):
                xx, yy = pos
                ival = int(round(val))
                if ival > Island.MAX_ENERGY:
                    ival = Island.MAX_ENERGY
                assert ival >= 0
                if self[pos]:
                    self._energy_map[yy][xx] = ival

            @property
            def energy_map(unused):
                return self._energy_map

        self._energy = _Energy()

    def __getitem__(self, pos):
        x, y = pos
        if 0 <= x < self.w and 0 <= y < self.h:
            return self._island_map[y][x]
        else:
            return False

    @staticmethod
    def init(island):
        return Island(island)

    @staticmethod
    def from_array(pieces):
        island = np.array(pieces[:, :, Configuration.ISLAND_IDX], dtype='>i4')
        energy = np.array(pieces[:, :, Configuration.ENERGY_IDX], dtype='>i4')
        return Island(island, energy)

    @staticmethod
    def encode_view(pieces, pose) -> None:
        horizon = 3
        horizon_map = list()
        dist = horizon
        for y in range(-dist, dist + 1):
            row = list()
            for x in range(-dist, dist + 1):
                row.append(distt((0, 0), (x, y)) <= horizon)
            horizon_map.append(row)

        row, col, enc = pieces.shape
        px, py = pose
        dist = horizon
        energy_map = np.copy(pieces[:, :, Configuration.ENERGY_IDX])
        pieces[:, :, Configuration.ENERGY_IDX] = 0
        for y in range(-dist, dist + 1):
            for x in range(-dist, dist + 1):
                if horizon_map[y + dist][x + dist] and 0 <= py + y < row and 0 <= px + x < col:
                    pieces[py + y, px + x, Configuration.ENERGY_IDX] = energy_map[py + y, px + x]

    @property
    def island_map(self):
        return self._island_map

    @property
    def energy(self):
        return self._energy
