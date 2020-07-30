from pygame.time import delay

from engine.botplayer import BotPlayer
from engine.config import GameConfig
from engine.game import Game
from engine.view import GameView
from gamer.p_lightgo.lightgo import LightGo

cfg_file = "maps/island.txt"
bots = [0]
DEBUG = False

config = GameConfig(cfg_file)
game = Game(config, len(bots))

actors = [
    BotPlayer(game, 0, LightGo(), debug=DEBUG),
]
for actor in actors:
    actor.initialize()

view = GameView(game)

coach = 0
while True:
    if view.close_event():
        break
    game.pre_round()
    view.update()
    for gamer in actors:
        gamer.turn()
        view.update()
    game.post_round()
    score = ""
    for i in range(len(bots)):
        score += " P%d: %d " % (i, game.players[i].score)
    print("########### ROUND %d SCORE: %s" % (coach, score))
    coach += 1
    delay(500)

view.update()
