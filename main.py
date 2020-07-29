import logging
import os

import coloredlogs

from TicTacToeGame import TicTacToeGame
from keras.NNetWrapper import NNetWrapper
from lib.Coach import Coach
from lib.utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

working_path = os.path.dirname(os.path.realpath(__file__))
args = dotdict({
    'numIters': 1000,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,
    'updateThreshold': 0.6,

    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': os.path.join(working_path, 'temp'),
    'load_model': False,
    'load_folder_file': (os.path.join(working_path, 'model'), 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', TicTacToeGame.__name__)
    g = TicTacToeGame(3)

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()