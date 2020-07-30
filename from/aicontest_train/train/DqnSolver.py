import os
import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.keras.models import load_model


class DQNSolver:
    LOAD = False
    SAVE_MODULE = "train/lh_go_model.h5"
    LOAD_MODULE = "train/best_lh_go_model.h5"
    VIEW_SCOPE = 3

    GAMMA = 0.80
    LEARNING_RATE = 0.001

    EXPLORATION_RATE = 1.0
    EXPLORATION_MIN = 0.01

    MEMORY_SIZE = 10000
    BATCH_SIZE = 10
    MAX_EPISODE = 500

    MAX_DISTANCE = 5
    REWARD_LH = 10
    REWARD_MISS_LEADING = -1
    REWARD_MISS_ACTION = -10
    REWARD_MISS_SESSION = -100
    REWARD_TERMINAL = 0

    def __init__(self, observation_space, action_space):
        self.exploration_rate = self.EXPLORATION_RATE

        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.buffer_memory = 0

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_move_mask, default_action) -> (int, str):
        if np.random.rand() < self.exploration_rate:
            action_id = self._random_valid_move(valid_move_mask)
            action_label = "rd"
        else:
            q_values = self.model.predict(state)
            action_id = int(np.argmax(q_values[0]))
            if valid_move_mask[action_id] == 0:
                action_id = default_action
            action_label = "nn"

        return action_label, action_id

    @staticmethod
    def _random_valid_move(valid_move_mask):
        valid_move_index = list()
        for idx, allowed_move in enumerate(valid_move_mask):
            if allowed_move == 1:
                valid_move_index.append(idx)
        random.shuffle(valid_move_index)
        action = valid_move_index.pop()
        return action

    def load_mode(self):
        if os.path.exists(self.LOAD_MODULE):
            self.model = load_model(self.LOAD_MODULE)

    def save_model(self, _id):
        # serialize weights to HDF5
        file_name, file_extension = os.path.splitext(self.SAVE_MODULE)
        model_path = file_name + "_" + str(_id) + file_extension
        self.model.save(model_path)

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        else:
            batch = random.sample(self.memory, self.BATCH_SIZE)
            for state, action, reward, state_next, terminal in batch:
                q_update = reward
                if not terminal:
                    n_q_values = self.model.predict(state_next)[0]
                    q_update = (reward + self.GAMMA * np.amax(n_q_values))
                q_values = self.model.predict(state)
                q_values[0][action] = q_update
                self.model.fit(state, q_values, verbose=0)

    def next_run(self, decay):
        self.exploration_rate *= decay
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
