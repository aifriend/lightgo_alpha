import math

import pygame

CELL = 15

TEXT = [
    (255, 255, 255)
]

PLAYERC = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 127, 0),
    (255, 127, 127),
]


class GameView(object):
    FPS = 100

    def __init__(self, game):
        self.game = game
        pygame.init()
        self.scale = 2
        self.fw = self.game.island.w * CELL * self.scale
        self.fh = self.game.island.h * CELL * self.scale
        size = width, height = self.fw, self.fh
        self.screen = pygame.display.set_mode(size)
        self.arena = pygame.Surface((self.fw, self.fh), 0, self.screen)
        self.view = pygame.Surface((self.fw, self.fh), 0, self.screen)
        self.view.set_alpha(100)
        self.nh = self.game.island.h - 1
        pygame.font.init()
        self.my_font_board = pygame.font.SysFont("Consolas", 9)
        self.my_font_label = pygame.font.SysFont("Consolas", 12)
        self.text_arena = ''

    def update(self, data=None):
        self.view.fill((0, 0, 0))
        self.arena.fill((0, 0, 0))
        for cy in range(self.game.island.h):
            for cx in range(self.game.island.w):
                if self.game.island[cx, cy]:
                    self.draw_cell((cx, cy))
        for (x0, y0), (x1, y1) in self.game.conns:
            owner = self.game.lighthouses[x0, y0].owner
            color = PLAYERC[owner]
            y0, y1 = self.nh - y0, self.nh - y1
            self._aaline((x0 * CELL + CELL / 2, y0 * CELL + CELL / 2),
                         (x1 * CELL + CELL / 2, y1 * CELL + CELL / 2), color)
        if isinstance(data, DataView):
            data.to_show(self)
        self._view()
        self.screen.blit(self.arena, (0, 0))
        self.screen.blit(self.view, (0, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(self.FPS)

    @staticmethod
    def quit():
        pygame.display.quit()

    def _afill(self, xy, wh, c):
        x0, y0 = xy
        w, h = wh
        x0 *= self.scale
        y0 *= self.scale
        w *= self.scale
        h *= self.scale
        self.arena.fill(c, (x0, y0, w, h))

    def _aaline(self, xy, x1y1, c):
        x0, y0 = xy
        x1, y1 = x1y1
        x0 *= self.scale
        y0 *= self.scale
        x1 *= self.scale
        y1 *= self.scale
        for i in range(self.scale):
            pygame.draw.aaline(self.arena, c, (x0 + i, y0 + i), (x1 + i, y1 + i))

    def _diamond(self, cxcy, size, c, width=0):
        cx, cy = cxcy
        cx *= self.scale
        cy *= self.scale
        size *= self.scale
        points = [
            (cx - size, cy),
            (cx, cy - size),
            (cx + size, cy),
            (cx, cy + size),
        ]
        pygame.draw.polygon(self.arena, c, points, width)

    def _to_text_board(self, content, pose):
        x0, y0 = pose
        x0 *= self.scale
        y0 *= self.scale
        self.text_arena = self.my_font_board.render(content, True, TEXT[0])
        self.arena.blit(self.text_arena, (x0, y0))

    def to_text_label(self, content, pose):
        x0, y0 = pose
        x0 *= self.scale
        y0 *= self.scale
        self.text_arena = self.my_font_label.render(content, True, TEXT[0])
        self.arena.blit(self.text_arena, (x0, y0))

    def _view(self):
        dist = self.game.island.HORIZON
        for cy in range(self.game.island.h):
            for cx in range(self.game.island.w):
                if self.game.island[cx, cy]:
                    py = (self.nh - cy) * CELL
                    px = cx * CELL
                    c_players = [i for i in self.game.players if i.pos == (cx, cy)]
                    if c_players:
                        player = c_players.pop()
                        color = self.cmul(PLAYERC[2], 0.3)
                        for y in range(-dist, dist + 1):
                            oy = y * CELL
                            for x in range(-dist, dist + 1):
                                ox = x * CELL
                                x0, y0 = (px + ox, py + oy)
                                w, h = (CELL, CELL)
                                x0 *= self.scale
                                y0 *= self.scale
                                w *= self.scale
                                h *= self.scale
                                self.view.fill(color, (x0, y0, w, h))

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

    def draw_cell(self, cxcy):
        cx, cy = cxcy
        py = (self.nh - cy) * CELL
        px = cx * CELL
        energy = self.game.island.energy[cx, cy]
        c = int(energy / 100.0 * 25)
        bg = tuple(map(int, (25 + c * 0.8, 25 + c * 0.8, 25 + c)))

        for vertices, fill in self.game.tris.items():
            if (cx, cy) in fill:
                owner = self.game.lighthouses[vertices[0]].owner
                bg = self.calpha(bg, PLAYERC[owner], 0.15)

        self._afill((px, py), (CELL, CELL), bg)
        self._afill((px + CELL / 2, py + CELL / 2), (1, 1), (255, 255, 255))
        self._to_text_board(str(energy), (px, py))

        c_players = [i for i in self.game.players if i.pos == (cx, cy)]
        if c_players:
            nx = int(math.ceil(math.sqrt(len(c_players))))
            wx = 12 / nx
            ny = int(math.ceil(len(c_players) / float(nx)))
            wy = 12 / ny
            for i, player in enumerate(c_players):
                iy = i / nx
                ix = i % nx
                color = self.cmul(PLAYERC[player.num], 0.5)
                self._afill((px + 2 + ix * wx, py + 2 + iy * wy),
                            (wx - 1, wy - 1), color)

        if (cx, cy) in self.game.lighthouses:
            lh = self.game.lighthouses[cx, cy]
            color = (192, 192, 192)
            if lh.owner is not None:
                color = PLAYERC[lh.owner]
            self._diamond((px + CELL / 2, py + CELL / 2), 4, color, 0)

    @staticmethod
    def close_event() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # This would be a quit event.
                return True  # So the user can close the program
        return False


class DataView:
    def __init__(self):
        self.off_set = 5
        self.args = dict()

    def update(self, *args):
        self.args = args

    def to_show(self, view: GameView):
        for idx, arg in enumerate(self.args):
            label, value = list(dict(arg).items()).pop()
            if isinstance(value, str):
                value_label = value
            elif isinstance(value, tuple):
                value_label = str(value)
            else:
                value_label = round(value, 3)
            view.to_text_label(
                f"{label}: {value_label}", (105, 1 + (idx * self.off_set)))
