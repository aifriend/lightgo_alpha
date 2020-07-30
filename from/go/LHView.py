import math
from typing import Any, Tuple, Optional

import numpy as np
import pygame

from examples.lh.config.config import CONFIG
from examples.lh.config.configuration import Configuration
from examples.lh.logic.board.board import Board
from lib.View import View

CELL = 15

PLAYERC = [
    (0, 255, 0),  # green
    (255, 0, 0),  # red
    (0, 255, 255),  # sky
    (255, 0, 255),  # pink
    (255, 127, 0),  # orange
    (255, 127, 127),  # skin
    (192, 192, 192),  # grey
]


class LHView(View):
    def __init__(self, game):
        self._logic = game.logic
        self._screen_write = 0
        self._scale = 0
        self._fw = 0
        self._fh = 0
        self._nw = 0
        self._nh = 0
        self._screen = None
        self._arena = None
        self._clock = None

    @property
    def scale(self):
        return self._scale

    @staticmethod
    def from_array(board):
        logic = Board()
        logic.from_array(board)
        return logic

    def initView(self, board) -> Optional[Tuple[Any, Any, Any]]:
        if not CONFIG.visibility:
            return

        self._scale = 2
        scree_size = width, height = 600, 640
        self._screen_write = width, height, offset = 600, 100, 540
        self._fw = self._logic.board.size[1] * CELL * self._scale  # 21 -> 630
        self._fh = self._logic.board.size[0] * CELL * self._scale  # 18 -> 540
        self._nw = self._logic.board.size[1] - 1  # 20
        self._nh = self._logic.board.size[0] - 1  # 17

        if CONFIG.visibility > 3:
            pygame.init()

            # screen
            self._screen = pygame.display.set_mode(scree_size)
            pygame.display.set_caption('Lighthouse Game')

            # surface
            # self._arena = pygame.Surface((self._fw, self._fh), 0, self._screen)
            self._arena = pygame.Surface(scree_size, 0, self._screen)  # enclosed window

            # clock
            self._clock = pygame.time.Clock()
        else:
            self._output(scree_size, self._logic.board)

    def display(self, board) -> Optional[Tuple[Any, Any, Any]]:
        """
        Console presentation of board

        :param board: game state
        :return: /
        """
        if not CONFIG.visibility:
            return

        if CONFIG.visibility > 3 and \
                not (self._arena is None) and \
                not (self._screen is None) and \
                not (self._clock is None):
            is_view_updated = self._update_graphics(board, Configuration.FPS)
            if not is_view_updated:
                self._output(board.shape, board)
                exit(1)

        return self._arena, self._screen, self._clock

    def _update_graphics(self, board: np.ndarray, fps: int = 100) -> bool:
        """
        Executes game tick on canvas, redrawing whole game state. Values here are somewhat hardcoded, which can be changed
        to display game in some nicer config.
        LHLogic size 8x8 is working best with this config, 6x6 might work as well, but other might not.
        :param board: game state that will be drawn
        :param fps: how many fps should pygame draw. if value is set to higher number than your pc can handle, it will draw at max possible.
        """
        # clear display
        self._arena.fill((0, 0, 0))

        # draw grid and board
        for cy in range(self._logic.board.size[0]):
            for cx in range(self._logic.board.size[1]):
                if self._logic.board.island[cx, cy]:
                    self.draw_cell((cx, cy))
                else:
                    self.draw_n_cell((cx, cy))

        # connections
        if len(self._logic.board.connection.conns) > 0:
            for (x0, y0), (x1, y1) in self._logic.board.connection.conns:
                owner = self._logic.board.lighthouses[x0, y0].owner
                color = PLAYERC[(0 if owner == -1 else 1)]
                y0, y1 = self._nh - y0, self._nh - y1
                self._aaline((x0 * CELL + CELL / 2, y0 * CELL + CELL / 2),
                             (x1 * CELL + CELL / 2, y1 * CELL + CELL / 2),
                             color)

        # board stats
        logic = LHView.from_array(board)
        i = 1
        for player in logic.players.values():
            p_s_msg = "(player %d): %d" % (player.turn, player.score)
            self.message_display("Score " + p_s_msg, (5, i))
            p_e_msg = "(player %d): %d" % (player.turn, player.energy)
            self.message_display("Energy " + p_e_msg, (9, i))
            i += 1

        # timing remaining
        time_remaining = board[0][0][Configuration.TIME_IDX]
        self.message_display("Remaining:  " + str(time_remaining), (1, 1))
        self._screen.blit(self._arena, (0, 0))

        pygame.display.flip()

        self._clock.tick(fps)

        return not self._close_event_graphics()

    def draw_cell(self, cxcy):
        cx, cy = cxcy
        py = (self._nh - cy) * CELL
        px = cx * CELL
        c = int(self._logic.board.island.energy[cx, cy] / 100.0 * 25)
        bg = tuple(map(int, (25 + c * 0.8, 25 + c * 0.8, 25 + c)))

        # show polygon affected area
        """
        if len(self._logic.board.poligon.tris) > 0:
            for vertices, fill in self._logic.board.poligon.tris.items():
                if (cx, cy) in fill:
                    owner = self._logic.board.lighthouses[vertices[0]].owner
                    bg = self.calpha(bg, PLAYERC[(0 if owner == -1 else (1 if owner == 0 else 2))], 0.15)
        """

        # cells
        self._afill((px, py), (CELL, CELL), bg)
        self._afill((px + CELL / 2, py + CELL / 2), (1, 1), (255, 255, 255))

        # players
        cplayers = [p for p in self._logic.board.players.values() if p.pos == (cx, cy)]
        if cplayers:
            nx = int(math.ceil(math.sqrt(len(cplayers))))
            wx = 12 / nx
            ny = int(math.ceil(len(cplayers) / float(nx)))
            wy = 12 / ny
            for i, player in enumerate(cplayers):
                iy = i / nx
                ix = i % nx
                p_color = (0 if player.turn == -1 else 1)
                color = self.cmul(PLAYERC[p_color], 0.5)
                self._afill((px + 2 + ix * wx, py + 2 + iy * wy), (wx - 1, wy - 1), color)

        # lighthouses
        if (cx, cy) in self._logic.board.lighthouses:
            lh = self._logic.board.lighthouses[cx, cy]
            color = (255, 255, 0)
            if lh.owner != 0:
                color = PLAYERC[(0 if lh.owner == -1 else 1)]
            self._diamond((px + CELL / 2, py + CELL / 2), 4, color, 0)

    def draw_n_cell(self, cxcy):
        cx, cy = cxcy
        py = (self._nh - cy) * CELL
        px = cx * CELL

        self._afill((px, py), (CELL, CELL), (0, 76, 153))

    def get_cell_by(self, pose):
        surface_width, surface_height = pose
        s_width = int(surface_width // (CELL * self._scale))
        s_height = int(surface_height // (CELL * self._scale))
        return s_width, s_height

    @staticmethod
    def _output(size, board):
        row, col, enc = size
        for y in range(row):
            print('-' * (row * 10 + 1))
            for x in range(col):
                a_player = board[y][x][Configuration.P_NAME_IDX]
                if a_player == 1:
                    a_player = '+1'
                if a_player == -1:
                    a_player = '-1'
                if a_player == 0:
                    a_player = ' 0'
                print("|" + a_player + " * " + str(int(board[y][x][Configuration.A_TYPE_IDX])) + " ", end="")
            print("|")
        print('-' * (row * 10 + 1))

    def message_display(self, text, position, text_size=16, color=(255, 255, 255)) -> None:
        """
        Display text on pygame window.
        :param text: string text
        :param position: coordinates on canvas where text will be displayed
        :param text_size: ...
        :param color: (r,g,b) color
        """
        large_text = pygame.font.SysFont('arial', text_size)
        text_surf = large_text.render(u"" + text, True, color)
        self._arena.blit(text_surf,
                         ((30 * position[0]), ((21 * position[1]) + int(self._screen_write[2]))))

    @staticmethod
    def _close_event_graphics() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # This would be a quit event.
                return True  # So the user can close the program
        return False

    def _afill(self, xy, wh, c):
        x0, y0 = xy
        w, h = wh
        x0 *= self._scale
        y0 *= self._scale
        w *= self._scale
        h *= self._scale
        self._arena.fill(c, (x0, y0, w, h))

    def _aaline(self, xy, x1y1, c):
        x0, y0 = xy
        x1, y1 = x1y1
        x0 *= self._scale
        y0 *= self._scale
        x1 *= self._scale
        y1 *= self._scale
        for i in range(self._scale):
            pygame.draw.aaline(self._arena, c, (x0 + i, y0 + i), (x1 + i, y1 + i))

    def _diamond(self, cxcy, size, c, width=0):
        cx, cy = cxcy
        cx *= self._scale
        cy *= self._scale
        size *= self._scale
        points = [
            (cx - size, cy),
            (cx, cy - size),
            (cx + size, cy),
            (cx, cy + size),
        ]
        pygame.draw.polygon(self._arena, c, points, width)

    @staticmethod
    def cmul(rgb, mul):
        r, g, b = rgb
        return int(r * mul), int(g * mul), int(b * mul)

    @staticmethod
    def calpha(r1g1b1, r2g2b2, a):
        r1, g1, b1 = r1g1b1
        r2, g2, b2 = r2g2b2
        return (int(r2 * a + r1 * (1 - a)),
                int(g2 * a + g1 * (1 - a)),
                int(b2 * a + b1 * (1 - a)))
