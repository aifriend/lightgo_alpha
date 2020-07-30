from typing import Tuple

import numpy as np

from examples.lh.config.encoders import OneHotEncoder
from lib.utils import dotdict


class LHConfig:
    class _NNetArgs:
        def __init__(self,
                     lr,
                     dropout,
                     epochs,
                     batch_size,
                     cuda,
                     num_channels):
            self.lr = lr  # learning rate
            self.dropout = dropout
            self.epochs = epochs  # times training examples are iterated through learning process
            self.batch_size = batch_size  # how many train examples are taken together for learning
            self.cuda = cuda  # this is only relevant when using TF GPU
            self.num_channels = num_channels  # used by nnet conv layers
            self.encoder = OneHotEncoder()

    class _GameConfig:
        def __init__(self,
                     initial_energy,
                     damage,
                     acts_enabled,
                     timeout):
            self.encoder = OneHotEncoder()

            # ##################################
            # ############# ENERGY #############
            # ##################################

            # how much initial energy do players get at game begining
            self.INITIAL_ENERGY = initial_energy

            # ##################################
            # ########### TIMEOUT ##############
            # ##################################

            # If timeout should be used. This causes game to finish after TIMEOUT number of actions.
            # Check if timeout is being used. Alternatively Kill function is used
            # how many turns until game end - this gets reduced when each turn is executed
            self.TIMEOUT = timeout

            # ##################################
            # ########## ATTACKING #############
            # ##################################

            # how much damage is dealt to attacked actor
            self.DAMAGE = damage

            self.acts_enabled = dotdict(acts_enabled or {
                "pass": True,
                "up": True,
                "down": True,
                "right": True,
                "left": True,
                "upright": True,
                "upleft": True,
                "downright": True,
                "downleft": True,
                "attack10": True,
                "attack30": True,
                "attack60": True,
                "attack80": True,
                "attack100": True,
                "connect0": True,
                "connect1": True,
                "connect2": True,
                "connect3": True,
                "connect4": True
            })

    class _PitArgs:
        def __init__(self,
                     player1_type,
                     player2_type,
                     player1_config,
                     player2_config,
                     player1_model_file,
                     player2_model_file,
                     num_games):

            self.player1_type = player1_type
            self.player2_type = player2_type
            self.player1_model_file = player1_model_file
            self.player2_model_file = player2_model_file
            self.player1_config = player1_config or {'numMCTSSims': 2, 'cpuct': 1.0}
            self.player2_config = player2_config or {'numMCTSSims': 2, 'cpuct': 1.0}
            self.num_games = num_games

        def create_players(self, game):
            return self._create_player(game, self.player1_type, self.player1_config, self.player1_model_file), \
                   self._create_player(game, self.player2_type, self.player2_config, self.player2_model_file)

        def _create_player(self,
                           game,
                           player_type: str,
                           player_config: dict,
                           player_model_file: str):
            from examples.lh.LHPlayers import RandomLHPlayer, GreedyLHPlayer, HumanLHPlayer

            if player_type == 'nnet':
                if player_config is None:
                    print("Invalid pit configuration. Returning")
                    exit(1)
                return self._PitNNetPlayer(game, player_config, player_model_file).play
            if player_type == 'random':
                return RandomLHPlayer(game).play
            if player_type == 'greedy':
                return GreedyLHPlayer(game).play
            if player_type == 'human':
                return HumanLHPlayer(game).play
            print("Invalid player type. Returning")
            exit(1)

        class _PitNNetPlayer:
            def __init__(self,
                         g,
                         player_config,
                         player_model_file):
                from examples.lh import NNetWrapper as NNet
                from lib.MCTS import MCTS

                n1 = NNet(g, OneHotEncoder())
                n1.load_checkpoint('./temp/', player_model_file)
                args1 = dotdict(player_config or {'numMCTSSims': 2, 'cpuct': 1.0})
                mcts1 = MCTS(g, n1, args1)
                self.play = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    class _LearnArgs:
        def __init__(self,
                     num_iters,
                     num_eps,
                     temp_threshold,
                     update_threshold,
                     maxlen_of_queue,
                     num_mcts_sims,
                     arena_compare,
                     cpuct,
                     checkpoint,
                     load_model,
                     load_folder_file,
                     num_iters_for_train_examples_history,
                     save_train_examples,
                     load_train_examples,
                     timeout):
            self.numIters = num_iters  # total number of games played from start to finish is numIters * numEps
            self.numEps = num_eps  # How may game is played in this episode
            self.tempThreshold = temp_threshold
            self.updateThreshold = update_threshold  # Percentage that new model has to surpass by win rate to replace old model
            self.maxlenOfQueue = maxlen_of_queue
            self.numMCTSSims = num_mcts_sims  # How many MCTS tree searches are performing (mind that this MCTS doesnt use simulations)
            self.arenaCompare = arena_compare  # How many comparisons are made between old and new model
            self.cpuct = cpuct  # search parameter for MCTS

            self.checkpoint = checkpoint
            self.load_model = load_model  # Load training examples from file - WARNING - this is disabled in LHPlayers.py because of memory errors received when loading data from file
            self.load_folder_file = load_folder_file
            self.numItersForTrainExamplesHistory = num_iters_for_train_examples_history  # maximum number of 'iterations' that game episodes are kept in queue. After that last is popped and new one is added.

            self.save_train_examples = save_train_examples
            self.load_train_examples = load_train_examples
            self.game_timeout = timeout

    def __init__(self,
                 learn_visibility=4,
                 pit_visibility=4,
                 timeout_player=200,

                 initial_energy_player1: int = 0,
                 damage_player1: int = 0,
                 acts_enabled_player1: dict = None,
                 player1_model_file: str = "best_player1.pth.tar",

                 initial_energy_player2: int = 0,
                 damage_player2: int = 0,
                 acts_enabled_player2: dict = None,
                 player2_model_file: str = "best_player2.pth.tar",

                 num_iters: int = 4,
                 num_eps: int = 4,
                 temp_threshold: int = 100,
                 update_threshold: float = 0.6,
                 maxlen_of_queue: int = 6400,
                 num_mcts_sims: int = 10,
                 arena_compare: int = 10,
                 cpuct: float = 1.41,  # sqrt(2) - MCTS(exploration/explotation)
                 checkpoint: str = './temp/',
                 load_model: bool = False,
                 load_folder_file: Tuple[str, str] = ('./temp/', 'checkpoint_0.pth.tar'),
                 num_iters_for_train_examples_history: int = 8,
                 save_train_examples: bool = False,
                 load_train_examples: bool = False,

                 # nnet, greedy, human, random
                 player1_type: str = 'random',
                 player2_type: str = 'random',
                 player1_config: dict = None,
                 player2_config: dict = None,
                 num_games: int = 4,

                 lr: float = 0.01,
                 dropout: float = 0.3,
                 epochs: int = 30,
                 batch_size: int = 256,
                 cuda: bool = True,
                 num_channels: int = 128
                 ):
        """
        :param learn_visibility: How much console should output while running learn. If visibility.verbose > 3, Pygame is shown
        :param pit_visibility: How much console should output while running pit. If visibility.verbose > 3, Pygame is shown
        :param timeout_player: After what time game will timeout
        :param initial_energy_player1: How much initial energy should player have
        :param damage_player1: How much damage is inflicted upon action 'attack' on other actor
        :param acts_enabled_player1: dictionary of which actions are enabled for player. See its default values to override.
        :param player1_model_file: Filename in temp folder that player 1 nnet player uses
        :param initial_energy_player2: How much initial energy should player have
        :param damage_player2: How much damage is inflicted upon action 'attack' on other actor
        :param acts_enabled_player2: dictionary of which actions are enabled for player. See its default values to override
        :param player2_model_file: Filename in temp folder that player 2 nnet player uses
        :param num_iters: How many iterations of games it should be played
        :param num_eps: How many episodes in each game iteration it should be played
        :param temp_threshold: Used by coach. "It uses a temp=1 if episodeStep < tempThreshold, and thereafter uses temp=0."
        :param update_threshold: Percentage of how much wins should newer model have to be accepted
        :param maxlen_of_queue: How many train examples can be stored in each iteration
        :param num_mcts_sims: How many MCTS sims are executed in each game episode while learning
        :param arena_compare: How many comparations of newer and older model should be made before evaluating which is better
        :param cpuct: Exploration parameter for MCTS
        :param checkpoint: folder where checkpoints should be saved while learning
        :param load_model: If model is loaded from checkpoint on learning start
        :param load_folder_file: tuple(folder, file) where model is loaded from
        :param num_iters_for_train_examples_history: How many iterations of train examples should be kept for learning. If this number is exceeded, oldest iteration of train exaples is removed from queue
        :param save_train_examples: If train examples should be saved to file (Caution if choosing this, because of memory error)
        :param load_train_examples: If train examples should be loaded from file (Caution if choosing this, because of memory error)
        :param player1_type: What type should player 1 be ("nnet", "random", "greedy", "human")
        :param player2_type: What type should player 2 be ("nnet", "random", "greedy", "human")
        :param player1_config: If "nnet" player is chosen, config can be provided {'numMCTSSims': 2, 'cpuct': 1.0}
        :param player2_config: If "nnet" player is chosen, config can be provided {'numMCTSSims': 2, 'cpuct': 1.0}
        :param num_games: How many games should be played for pit config
        :param lr: Learning rate of model
        :param dropout: Dropout in NNet Model config
        :param epochs: How many epochs should learning take
        :param batch_size: How big batches of learning examples there should be while learning
        :param cuda: Whether to use cuda if tensorflow gpu is installed and GPU supports cuda operations
        :param num_channels: Number of channels in NNet Model config

        """

        # output for game stats during playing games (game_episode, game iteration, player name, action executed,
        # action_name, action_direction, player_score...
        self.config_file_pit = "./temp/config_pit.csv"
        self.config_file_learn = "./temp/config_learn.csv"

        self.board_file_path = "./config/maps/island.txt"

        self.visibility = 4
        self._pit_visibility = pit_visibility
        self._learn_visibility = learn_visibility
        self.timeout = timeout_player

        self.player1_config = self._GameConfig(
            initial_energy=initial_energy_player1,
            damage=damage_player1,
            acts_enabled=acts_enabled_player1,
            timeout=timeout_player)

        self.player2_config = self._GameConfig(
            initial_energy=initial_energy_player2,
            damage=damage_player2,
            acts_enabled=acts_enabled_player2,
            timeout=timeout_player)

        self.learn_args = self._LearnArgs(
            num_iters=num_iters,
            num_eps=num_eps,
            temp_threshold=temp_threshold,
            update_threshold=update_threshold,
            maxlen_of_queue=maxlen_of_queue,
            num_mcts_sims=num_mcts_sims,
            arena_compare=arena_compare,
            cpuct=cpuct,
            checkpoint=checkpoint,
            load_model=load_model,
            load_folder_file=load_folder_file,
            num_iters_for_train_examples_history=num_iters_for_train_examples_history,
            save_train_examples=save_train_examples,
            load_train_examples=load_train_examples,
            timeout=timeout_player)

        self.pit_args = self._PitArgs(
            player1_type=player1_type,
            player2_type=player2_type,
            player1_config=player1_config,
            player2_config=player2_config,
            player1_model_file=player1_model_file,
            player2_model_file=player2_model_file,
            num_games=num_games
        )

        self.nnet_args = self._NNetArgs(
            lr=lr,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            num_channels=num_channels
        )

    def set_runner(self, runner: str):
        if runner == 'pit':
            self.visibility = self._pit_visibility
        elif runner == 'learn':
            self.visibility = self._learn_visibility
        else:
            print("Unrecognised runner. Returning")
            exit(1)

    def set_board(self, path_to_board=None):
        self.board_file_path = path_to_board
