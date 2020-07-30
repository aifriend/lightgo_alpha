"""
Compares 2 players against each other and outputs num wins p1/ num wins p2/ draws

"""
from examples.lh.LHGame import LHGame
from examples.lh.LHView import LHView
from examples.lh.config.config import CONFIG
from lib import Arena

CONFIG.set_runner('pit')

g = LHGame()
v = LHView(g)

player1, player2 = CONFIG.pit_args.create_players(g)

arena = Arena.Arena(player1, player2, g, v)
print(arena.playGames(CONFIG.pit_args.num_games, verbose=CONFIG.visibility))
