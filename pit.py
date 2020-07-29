"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
import numpy as np

from TicTacToeGame import TicTacToeGame
from TicTacToePlayers import RandomPlayer, HumanTicTacToePlayer, GreedyTicTacToePlayer
from keras.NNetWrapper import NNetWrapper
from lib import Arena
from lib.MCTS import MCTS
from lib.utils import dotdict

human_vs_cpu = True

g = TicTacToeGame(3)

# all players
rp = RandomPlayer(g).play
gp = GreedyTicTacToePlayer(g).play
hp = HumanTicTacToePlayer(g).play

# nnet players
n1 = NNetWrapper(g)
n1.load_checkpoint('model/', 'best-25eps-25sim-10epch.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNetWrapper(g)
    n2.load_checkpoint('model/', 'best-25eps-25sim-10epch.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=TicTacToeGame.display)

print(arena.playGames(20, verbose=True))
