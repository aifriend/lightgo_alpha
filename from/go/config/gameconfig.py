class MoveError(Exception):
    pass


class GameError(Exception):
    pass


class CommError(Exception):
    pass


class GameConfig(object):
    def __init__(self, map_file):
        with open(map_file, "r") as fd:
            lines = [l.replace("\n", "") for l in fd.readlines()]

        self._lighthouses = list()
        self._island = list()

        # actors
        players = list()
        for y, line in enumerate(lines[::-1]):
            row = list()
            for x, c in enumerate(line):
                if c == "#":
                    row.append(0)
                elif c == "!":
                    row.append(1)
                    self._lighthouses.append((x, y))  # lighthouses
                elif c == " ":
                    row.append(1)
                else:
                    row.append(1)
                    players.append((c, (x, y)))  # players
            self._island.append(row)

        # players
        self._players = [(c, pos) for c, pos in sorted(players)]

        # island
        self._width = len(self._island[0])
        self._height = len(self._island)
        if not all(len(l) == self._width for l in self._island):
            raise GameError("All map rows must have the same width")
        if (not all(not i for i in self._island[0]) or
                not all(not i for i in self._island[-1]) or
                not all(not (i[0] or i[-1]) for i in self._island)):
            raise GameError("Map border must not be part of island")

    @property
    def size(self):
        return self._height, self._width

    @property
    def f_island(self):
        return self._island

    @property
    def f_players(self):
        return self._players

    @property
    def f_lighthouses(self):
        return self._lighthouses
