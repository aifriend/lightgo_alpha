import numpy as np

from examples.lh.config.configuration import Configuration
from examples.lh.config.gameconfig import MoveError
from examples.lh.logic.board.board import Board


class LHLogic:
    """
    Defines game rules (action checking, end-game conditions)
    can_execute_move is checking if move can be executed and execute_move is applying this move to new board
    """

    def __init__(self, cfg_file) -> None:
        self.config_file = cfg_file
        self._board = Board()

    @property
    def board(self):
        return self._board

    def initialize(self):
        self._board.init(self.config_file)

    def from_array(self, board):
        self._board = Board()
        self._board.from_array(board)

    def to_array(self, board):
        return self._board.to_array(board)

    def get_next_state(self, turn, action, board):
        """
        Player 1 trigger pre_round pre-processing
        Player -1 trigger post_round post-processing
        """
        # get player
        player = self._board.player_by(turn)
        if not player:
            return

        # Execute move
        row, col = self._board.size
        y, x, action_index = np.unravel_index(action, [row, col, Configuration.NUM_ACTS])
        act = Configuration.ACTS_REV[action_index]
        act_value = self.get_valid_actions(action_index, turn)
        if not act_value:
            raise MoveError("Not action value for %r" % action_index)

        if act == "pass":
            return

        # Game player move
        elif act == "up" or act == "down" or act == "right" or act == "left" or \
                act == "upright" or act == "upleft" or act == "downright" or act == "downleft":
            player.move(board[:, :, Configuration.ISLAND_IDX], act_value)

        # Game LH update
        elif act == "attack100" or act == "attack80" or act == "attack60" or \
                act == "attack30" or act == "attack10":
            if not isinstance(act_value, int):
                raise MoveError("Attack command requires integer energy")
            if player.pos not in self._board.lighthouses:
                raise MoveError("Player must be located at target lighthouse")
            self._board.lighthouses[player.pos].attack(player, act_value)

        # Game connection
        elif act == "connect0" or act == "connect1" or act == "connect2" or \
                act == "connect3" or act == "connect4":
            if not isinstance(act_value, tuple):
                raise MoveError("Connect command requires destination")
            try:
                dest = act_value
                hash(dest)
            except Exception:
                raise MoveError("Destination must be a coordinate pair")
            self._board.connect(player, dest)

        else:
            raise MoveError("Invalid command %r with param %r" % (act, act_value))

    def get_valid_actions(self, move, turn):
        # get player
        player = self._board.player_by(turn)
        if not player:
            return None

        if min(Board.POSSIBLE_MOVES) <= move <= max(Board.POSSIBLE_MOVES):
            action = self._board.get_available_worker_actions(player, move)
            if action:
                return action

        elif min(Board.POSSIBLE_ATTACK) <= move <= max(Board.POSSIBLE_ATTACK):
            action = self._board.get_available_attack_actions(player, move)
            if action:
                return action

        elif min(Board.POSSIBLE_CONNECTIONS) <= move <= max(Board.POSSIBLE_CONNECTIONS):
            action = self._board.get_available_lh_connection_actions(player, move)
            if action:
                return action

        # No action found to be returned
        assert False

    def get_valid_moves(self, turn):
        # get player
        player = self._board.player_by(turn)
        if not player:
            return None

        # get valid moves
        moves = []

        # select valid moves
        for row in range(self._board.size[0]):
            for col in range(self._board.size[1]):
                if (col, row) == player.pos:
                    valid_moves = [0] * Configuration.NUM_ACTS

                    # AVAILABLE ACTION - WK - MOVE
                    self._board.get_available_worker_moves(player, valid_moves)

                    # AVAILABLE ACTION - WK - ATTACK
                    self._board.get_available_attack_moves(player, valid_moves)

                    # AVAILABLE ACTION - LH - CONN
                    self._board.get_available_lh_connection_moves(player, valid_moves)

                    moves.extend(valid_moves)
                else:
                    moves.extend([0] * Configuration.NUM_ACTS)

        # return the generated move list
        return moves

    def valid_moves_left(self, turn):
        # get player
        player = self._board.player_by(turn)
        if not player:
            return None

        # AVAILABLE ACTION - WK - MOVE
        if next(self._board.available_worker_moves(player)):
            return True

        # AVAILABLE ACTION - WK - ATTACK
        elif next(self._board.available_attack_moves(player)):
            return True

        # AVAILABLE ACTION - LH - CONN
        elif next(self._board.available_lh_connection_moves(player)):
            return True

        # return the generated move list
        return False
