from copy import deepcopy

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from botplayer import BotPlayer
from lightgo import LightGo
from train.DqnSolver import DQNSolver


class CoachService:
    class ActionSpace:
        ACTION_PASS = 4  # pass
        ACTION_SPACE = 9  # removed attack and connect
        ENERGY_PLAYER_POSE = 25

        MOVE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ATTACK = [9]
        CONNECT = [10]

        POSSIBLE_MOVES = {
            0: (-1, -1),  # down left
            1: (-1, 0),  # left
            2: (-1, 1),  # up left
            3: (0, -1),  # down
            ACTION_PASS: (0, 0),  # pass
            5: (0, 1),  # up
            6: (1, -1),  # down right
            7: (1, 0),  # right
            8: (1, 1),  # up right
        }

        @staticmethod
        def get_action_id(delta):
            for k, v in CoachService.ActionSpace.POSSIBLE_MOVES.items():
                if v == delta:
                    return k
            return -1

        @staticmethod
        def action_translation(action_id: int) -> (str, tuple):
            action_type = "move"
            pose = CoachService.ActionSpace.POSSIBLE_MOVES[action_id]
            if action_id not in CoachService.ActionSpace.MOVE:
                raise Exception("No action allowed!")
            return action_type, pose

    def __init__(self, bot_player: BotPlayer):
        self.game = bot_player.game
        self._bot_player = bot_player

        self.BOARD_DIM = 2
        self.VIEW_DIM = 49
        self.VIEW_SCOPE = 7
        self.PLAYER_ENERGY_DIM = 1
        self.PLAYER_MAX_ENERGY = 10000

        self.PLAYER_START_AT = 0
        self.PLAYER_END_AT = self.PLAYER_START_AT + self.BOARD_DIM
        self.PLAYER_ENERGY_START_AT = self.PLAYER_END_AT
        self.PLAYER_ENERGY_END_AT = self.PLAYER_ENERGY_START_AT + self.PLAYER_ENERGY_DIM
        self.LH_START_AT = self.PLAYER_ENERGY_END_AT
        self.LH_END_AT = self.LH_START_AT + len(self.game.lighthouses) * self.BOARD_DIM
        self.LH_HAVE_KEY_START_AT = self.LH_END_AT
        self.LH_HAVE_KEY_END_AT = self.LH_HAVE_KEY_START_AT + len(self.game.lighthouses)
        self.LH_DIST_START_AT = self.LH_HAVE_KEY_END_AT
        self.LH_DIST_END_AT = self.LH_DIST_START_AT + len(self.game.lighthouses)
        self.ACTION_MOVE_START_AT = self.LH_DIST_END_AT
        self.ACTION_MOVE_END_AT = self.ACTION_MOVE_START_AT + len(self.ActionSpace.MOVE)
        self.end_coding = self.ACTION_MOVE_END_AT

        self.scalar = self._get_normalizer()
        self.lh_target_list = list()
        self.lh_target = self._get_new_lh_target()

    def get_observation(self):
        _observation, _observation_raw = \
            self._game_to_state(self._bot_player)
        return _observation, _observation_raw

    def _game_to_state(self, bot_player):
        # update game base on last player move
        current_state = bot_player.get_state()

        # player pose
        _observation_raw = list(current_state["position"])

        # player energy
        _observation_raw += [current_state["energy"]]

        # light houses: pose, key and distance
        lh_poses, lh_keys, lh_dists = \
            self._lh_to_state(bot_player.gamer, current_state)
        _observation_raw += lh_poses
        _observation_raw += lh_keys
        _observation_raw += lh_dists

        # possible moves
        _observation_raw += \
            self._possible_mov_to_state(current_state)

        _observation = self.scalar.transform(np.reshape(
            _observation_raw, (1, self.get_observation_size())))

        return _observation, _observation_raw

    def get_next_observation(self):
        """
        update game base on last player move

        :return:
        """
        my_bot_player = deepcopy(self._bot_player)
        my_bot_player.game.pre_round()
        _observation, _observation_raw = self._game_to_state(my_bot_player)
        del my_bot_player

        return _observation, _observation_raw

    def _get_dist_to_target_lh(self):
        # closest lh distance from actor
        player_pose = self._bot_player.player.pos
        dist_to_target_lh = \
            self._bot_player.gamer.lh_service.get_lh_dist(
                player_pose, self.lh_target)
        return dist_to_target_lh[1]

    def _get_new_lh_target(self):
        player_pose = self._bot_player.player.pos
        lh_target = \
            self._bot_player.gamer.lh_service.get_closest_lh(
                player_pose, self.lh_target_list)
        if not len(lh_target):
            raise ValueError("No lighthouse left to target!")
        self.lh_target_list.append(lh_target[0])
        return lh_target[0]

    def get_reward(self, step, action, terminal):
        reward = DQNSolver.REWARD_MISS_LEADING

        if action == self.ActionSpace.ACTION_PASS:
            # miss action
            return DQNSolver.REWARD_MISS_ACTION
        elif step > DQNSolver.MAX_EPISODE:
            # terminal reward without success
            return DQNSolver.REWARD_MISS_SESSION
        elif terminal:
            return DQNSolver.REWARD_TERMINAL

        # reward by lh distance without key
        dist_to_target_lh = self._get_dist_to_target_lh()
        if self.lh_target not in self._bot_player.player.keys:
            if dist_to_target_lh == 0:
                reward = DQNSolver.REWARD_LH
            else:
                scope_lh_dist = DQNSolver.MAX_DISTANCE - dist_to_target_lh
                if scope_lh_dist >= 0:
                    reward = scope_lh_dist
                else:
                    reward = DQNSolver.REWARD_MISS_LEADING
        else:
            self.lh_target = self._get_new_lh_target()

        return reward

    def get_observation_size(self):
        return self.end_coding

    def is_terminal(self):
        return all(
            lh.pos in self._bot_player.player.keys
            for lh in self.game.lighthouses.values())

    def run_pre_round(self):
        self.game.pre_round()

    def run_post_round(self):
        self.game.post_round()

    def run_step(self, state, dqn_solver):
        pose = self._bot_player.player.pos
        valid_move_mask = self.get_valid_action(pose)
        action_label, action_id = dqn_solver.act(
            state, valid_move_mask, self.ActionSpace.ACTION_PASS)
        action_type, action_pose = self.translate_action(action_id)

        return action_label, action_id, action_type, action_pose

    def run_next_step(self, action_type: str, pose):
        try:
            if action_type == "move":
                # move to
                if isinstance(pose, tuple):
                    self._bot_player.player.move((pose[0], pose[1]))
                # light house owned when key is taken
                if self._bot_player.player.pos in self.game.lighthouses:
                    self.game.lighthouses[
                        self._bot_player.player.pos].owned(
                        self._bot_player.player)
            self._bot_player.gamer.success()
        except:
            pass

    def get_valid_action(self, pose, default=0) -> list:
        valid = [default] * self.ActionSpace.ACTION_SPACE
        legal_delta = \
            self._bot_player.gamer.motion_service.get_possible_moves(pose)
        if len(legal_delta) == 0:
            raise ValueError("No valid moves available!")
        for delta in legal_delta:
            valid[self.ActionSpace.get_action_id(delta)] = 1

        return valid

    @staticmethod
    def translate_action(action_id: int) -> (int, tuple):
        """
        Get motion type and offset from delta ID

        :param action_id:
        :return: action_label, delta
        """
        return CoachService.ActionSpace.action_translation(action_id)

    def _get_normalizer(self):
        scalar = MinMaxScaler(feature_range=(0, 1))

        # min data length
        min_fit_data = np.zeros(shape=(1, self.get_observation_size()))

        # max data length
        board_list = [self.game.island.w, self.game.island.h]
        n_lh = len(self.game.lighthouses)
        lh_dist_limit = np.empty(shape=(1, n_lh))
        lh_dist_limit.fill(float(max(
            self.game.island.w, self.game.island.h) + 1))
        max_fit_data = np.concatenate((
            np.array(board_list).reshape(1, self.BOARD_DIM),  # player pose
            np.array([self.PLAYER_MAX_ENERGY]).reshape(1, self.PLAYER_ENERGY_DIM),  # player energy
            np.array(board_list * n_lh).reshape(1, n_lh * self.BOARD_DIM),  # lh pose
            np.ones(shape=(1, n_lh)),  # lh keys
            lh_dist_limit,  # lh dist
            np.ones(shape=(1, len(self.ActionSpace.MOVE))),  # lh moves
        ), axis=1)
        scalar_fit_data = np.concatenate(
            (min_fit_data, max_fit_data)).reshape(2, max_fit_data.size)

        scalar.fit(scalar_fit_data)
        return scalar

    def _possible_mov_to_state(self, state: dict):
        return list(map(lambda x: float(x),
                        self.get_valid_action(state["position"])))

    @staticmethod
    def _lh_to_state(gamer: LightGo, state: dict):
        lh_states = gamer.lh_service.get_lh_states(state)

        lh_state_sorted = list(sorted(lh_states.items()))

        lh_pose_list = list()
        lh_key_list = list()
        lh_dist_list = list()
        for kv in lh_state_sorted:
            lh_pose_list.extend(kv[1]['position'])
            lh_key_list.append(int(kv[1]['have_key']))
            lh_dist_list.append(kv[1]['cur_dist'])

        return lh_pose_list, lh_key_list, lh_dist_list
