from examples.lh import NNetWrapper
from examples.lh.LHGame import LHGame
from examples.lh.LHView import LHView
from examples.lh.config.config import CONFIG
from lib.Coach import Coach

if __name__ == "__main__":
    """
    Teaches neural network playing of specified game configuration using self play
    This configuration needs to be kept seperate, as different nnet and game configs are set

    """
    CONFIG.set_runner('learn')

    # Board config file
    CONFIG.set_board("config/maps/island.txt")

    # Create nnet for this game
    g = LHGame()
    v = LHView(g)

    # Create network
    nnet = NNetWrapper(g, CONFIG.nnet_args.encoder)

    # If training examples should be loaded from file
    if CONFIG.learn_args.load_model:
        nnet.load_checkpoint(CONFIG.learn_args.load_folder_file[0], CONFIG.learn_args.load_folder_file[1])

    # Create coach instance that starts teaching nnet on newly created game using self-play
    c = Coach(g, nnet, CONFIG.learn_args, v)
    if CONFIG.learn_args.load_train_examples:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
