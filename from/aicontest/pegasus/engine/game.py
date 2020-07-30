#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from pygame.time import delay

import botplayer
import engine
import view

cfg_file = sys.argv[1]
bots = sys.argv[2:]
DEBUG = False

config = engine.GameConfig(cfg_file)
game = engine.Game(config, len(bots))
actors = [botplayer.BotPlayer(game, i, cmdline, debug=DEBUG) for i, cmdline in enumerate(bots)]

for actor in actors:
    actor.initialize()

window = view.GameView(game)

coach = 0
while True:
    if window.close_event():
        break
    game.pre_round()
    window.update()
    for actor in actors:
        actor.turn()
        window.update()
    game.post_round()
    score = ""
    for i in range(len(bots)):
        score += " P%d: %d" % (i, game.players[i].score)
    print "### ROUND %d SCORE %s\n" % (coach, score),
    coach += 1
    delay(200)

window.update()
