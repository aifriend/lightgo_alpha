"""
Contains 3 players (human player, random player, greedy player (if searching for nnet player, it is defined by pre-learnt model)
Human player has defined input controls for Pygame and console

"""
import sys

import numpy as np
import pygame

from examples.lh.config.config import CONFIG
from examples.lh.config.configuration import Configuration


class RandomLHPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        print("RandomLHPlayer")
        action = np.random.randint(self.game.getActionSize())
        valid = self.game.getValidMoves(board, 1)
        while valid[action] != 1:
            # select random action on present state
            action = np.random.randint(self.game.getActionSize())

        y, x, action_index = np.unravel_index(action, [board.shape[0], board.shape[1], Configuration.NUM_ACTS])
        print("RP > returned act: Col(%s) Row(%s) Action(%s) Index(%r:%r)" % (
            x, y, Configuration.ACTS_REV[action_index], action, action_index))

        return action


class GreedyLHPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)

        print("GreedyLHPlayer")
        print("GP > sum valids: %s", sum(valid))
        candidates = []
        for a in range(self.game.getActionSize()):
            if valid[a] == 0:
                continue
            next_board, _ = self.game.getNextState(board, 1, a)
            score = self.game.get_score(next_board, 1)
            candidates += [(-score, a)]

        # select best action on present state
        candidates.sort()
        action = candidates[0][1]
        y, x, action_index = np.unravel_index(action, [board.shape[0], board.shape[1], Configuration.NUM_ACTS])
        print("GP > > returned act: Col(%s) Row(%s) Action(%s) Index(%r:%r)" % (
            x, y, Configuration.ACTS_REV[action_index], action, action_index))

        return action


class HumanLHPlayer:
    def __init__(self, game) -> None:
        self.game = game

    def play(self, board: np.ndarray):
        print("HumanLHPlayer")
        tup = tuple()
        valid = self.game.getValidMoves(board, 1)
        valid_pose = self._display_valid_moves(board, valid)
        while True:
            if CONFIG.visibility > 3:
                a = HumanLHPlayer._manage_input()
                action_index = a
            else:
                a = (input('type one of above actions in "action_name" format\n')).split(" ")
                action_index = Configuration.ACTS[a]

            # parse input
            try:
                tup = (int(valid_pose[1]), int(valid_pose[0]), int(action_index))
                a = np.ravel_multi_index(tup, (board.shape[0], board.shape[1], Configuration.NUM_ACTS))
            except Exception as exc:
                print("Could not parse action")

            # convert to action index in valids array
            if valid[a]:
                break
            else:
                print('This action is invalid!')
                self._display_valid_moves(board, valid)

        print("HP > returned act: Col(%s) Row(%s) Action(%s) Index(%r:%r)" % (
            tup[0], tup[1], Configuration.ACTS_REV[tup[2]], a, action_index))

        return a

    @staticmethod
    def _display_valid_moves(board, valid):
        valid_pose = set()
        valid_actions = dict()
        for i in range(len(valid)):
            if valid[i]:
                y, x, action_index = np.unravel_index(i, [board.shape[0], board.shape[1], Configuration.NUM_ACTS])
                name_action = Configuration.ACTS_REV[action_index]
                shortcut_action = Configuration.d_user_shortcuts_rev[action_index]
                valid_actions[shortcut_action] = name_action
                valid_pose.add((x, y))
        print(', '.join([str(v) + "(" + str(k) + ")" for k, v in valid_actions.items()]))
        assert len(valid_pose) == 1
        return valid_pose.pop()

    @staticmethod
    def _manage_input():
        while True:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN:
                    try:
                        return Configuration.d_user_shortcuts[event.unicode]
                    except Exception as exc:
                        print("shortcut '" + event.unicode + "' not supported.")
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(1)
