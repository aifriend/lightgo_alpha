from typing import Tuple

import numpy as np

from examples.lh.LHLogic import LHLogic
from examples.lh.config.config import CONFIG
from examples.lh.config.configuration import Configuration
from lib.Game import Game


class LHGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.logic = LHLogic(CONFIG.board_file_path)
        self.logic.initialize()

    def getInitBoard(self) -> np.ndarray:
        """
        :return: Returns new board from initial_board_config. That config can be dynamically changed as game progresses.
        """
        self.logic.initialize()

        # board pieces initialization
        pieces = np.zeros((self.logic.board.size[0], self.logic.board.size[1], Configuration.NUM_ENCODERS))

        # remaining time is stored in all squares
        pieces[:, :, Configuration.TIME_IDX] = CONFIG.timeout

        # get pieces from actual board state
        pieces = self.logic.to_array(pieces)

        return pieces

    def getNextState(self, board: np.ndarray, player_id: int, move: int) -> Tuple[np.ndarray, int]:
        """
        Gets next state for board. It also updates tick for board as game tick iterations are transfered
        within board as the last encoded parameter

        :param board: current board
        :param player_id: player executing action
        :param move: move to apply to new board
        :return: new board with applied action
        """
        # update board status
        self.logic.from_array(board)

        # pre-round game player and lh energy update
        self.logic.board.pre_round()

        # player keys and energy update
        self.logic.board.pre_player_update(player_id)

        # execute move
        self.logic.get_next_state(player_id, move, board)

        # player score update
        self.logic.board.post_player_update(player_id)

        # update board pieces
        pieces = self.logic.to_array(board)

        # get config for timeout: update timer on every tile:
        pieces[:, :, Configuration.TIME_IDX] -= 1

        return pieces, -player_id

    def getValidMoves(self, board: np.ndarray, player_id: int) -> np.ndarray:
        # update board status
        self.logic.from_array(board)

        # on-the-fly pre-round game update
        self.logic.board.pre_round()

        # player game update
        self.logic.board.pre_player_update(player_id)

        # select valid moves
        valid = self.logic.get_valid_moves(player_id)

        return np.array(valid)

    def getGameEnded(self, board: np.ndarray, player_id) -> float:
        """
        It is hard to decide when to finish real time strategy game, as players might not have enough time
        to execute wanted actions, but in the other hand, if players are left to play for too long, games become
        very long, or even 'infinitely' long.
        Using timeout. Timeout just cuts game and evaluates winner using one of 3 elo functions. We've found this one
        to be more useful, as it can be applied in 3d real time strategy games easier and more sensibly.

        :param board: current game state
        :param player_id: current player
        :return: real number on interval [-1,1] - return 0 if not ended, 1 if player 1 won, -1 if player 1 lost, 0.001 if tie
        """
        # update board status
        self.logic.from_array(board)

        # detect timeout
        if board[0, 0, Configuration.TIME_IDX] < 1:
            score_player1 = self.get_score(board, player_id)
            score_player2 = self.get_score(board, -player_id)
            if score_player1 == score_player2:
                return 0.001
            better_player = 1 if score_player1 > score_player2 else -1
            return better_player

        # detect no valid actions
        if not self.logic.valid_moves_left(1):
            return -1

        if not self.logic.valid_moves_left(-1):
            return 1

        # continue game
        return 0

    def getBoardSize(self) -> Tuple[int, int, int]:
        return self.logic.board.size[0], self.logic.board.size[1], Configuration.NUM_ENCODERS

    def getActionSize(self) -> int:
        return self.logic.board.size[0] * self.logic.board.size[1] * Configuration.NUM_ACTS

    def getCanonicalForm(self, board: np.ndarray, player_id: int) -> np.ndarray:
        self.logic.from_array(board)

        # revert player
        for player in self.logic.board.players.values():
            player.turn = player.turn * player_id

        # revert lighthouse
        for lh in self.logic.board.lighthouses.values():
            lh.owner = lh.owner * player_id

        # get board pieces
        n_board = self.logic.board.to_array(board)

        return n_board

    def getSymmetries(self, board: np.ndarray, pi):
        row, col = self.logic.board.size
        assert (len(pi) == row * col * Configuration.NUM_ACTS)
        pi_board = np.reshape(pi, (row, col, Configuration.NUM_ACTS))
        # squared, mirror and rotational
        is_square = all(len(row) == len(board) for row in board)
        if is_square:
            return_list = []
            for i in range(1, 5):
                for j in [True, False]:
                    new_b = np.rot90(board, i)
                    new_pi = np.rot90(pi_board, i)
                    if j:
                        new_b = np.fliplr(new_b)
                        new_pi = np.fliplr(new_pi)
                    return_list += [(new_b, list(new_pi.ravel()))]
        else:
            return_list = []
            for i in range(2, 5, 2):
                for j in [True, False]:
                    new_b = np.rot90(board, i)
                    new_pi = np.rot90(pi_board, i)
                    if j:
                        new_b = np.fliplr(new_b)
                        new_pi = np.fliplr(new_pi)
                    return_list += [(new_b, list(new_pi.ravel()))]

        return return_list

    def stringRepresentation(self, board: np.ndarray) -> bytes:
        return board.tostring()

    def get_score(self, board: np.array, turn: int) -> int:
        """
        Uses an elo function that determine better player

        :param board: game state
        :param turn: current player
        :return: elo for current player on this board
        """
        # update board status
        self.logic.from_array(board)

        # get player
        player = self.logic.board.player_by(turn)
        opponent = self.logic.board.player_by(-turn)
        assert player and opponent

        return player.score - opponent.score
