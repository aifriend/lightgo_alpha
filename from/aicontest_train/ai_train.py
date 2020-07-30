import random

from pygame.time import delay

from engine.botplayer import BotPlayer
from engine.config import GameConfig
from engine.game import Game
from engine.view import GameView, DataView
from gamer.p_lightgo.lightgo import LightGo
from train.CoachService import CoachService
from train.DqnSolver import DQNSolver
from train.ScoreLogger import ScoreLogger


class Coach:
    cfg_file = "maps/island_train.txt"
    bots = [0]
    DEBUG = True
    LOAD = False
    PLAYER = 0
    GAME_DELAY = 0
    VIEW = True

    def __init__(self):
        self.config = GameConfig(self.cfg_file)
        self.score_logger = ScoreLogger()

    def initialize_game(self):
        game = Game(self.config, len(self.bots))

        # actors
        actors = [
            BotPlayer(game, self.PLAYER, LightGo(), debug=self.DEBUG),
        ]
        for actor in actors:
            actor.initialize()
            # random init pose
            pose = (0, 0)
            while not actor.gamer.motion_service.is_in_island(pose):
                pose = (
                    random.randint(1, game.island.w - 2),
                    random.randint(1, game.island.h - 2)
                )
            actor.player.pos = pose

        # only learn from main actor: PLAYER (LightGo)
        actor = actors[self.PLAYER]

        # coach service
        coach_service = CoachService(actor)
        return coach_service

    def get_rl_solver(self):
        # game
        coach_service = self.initialize_game()

        dqn_solver = DQNSolver(
            coach_service.get_observation_size(),
            coach_service.ActionSpace.ACTION_SPACE)
        del coach_service

        if self.LOAD:
            dqn_solver.load_mode()

        return dqn_solver

    def train(self):
        run = True
        view = None
        view_data = DataView()

        dqn_solver = self.get_rl_solver()

        coach_run = 0
        while run:
            coach_run += 1
            coach_service = self.initialize_game()

            # view
            if self.VIEW:
                if view is not None:
                    view.quit()
                view = GameView(coach_service.game)
                view.update()

            # step
            step = 0
            while run:
                step += 1
                if self.VIEW and view.close_event():
                    run = False

                # pre round
                coach_service.run_pre_round()
                if self.VIEW:
                    view.update()

                # actual state
                state, state_raw = coach_service.get_observation()
                # next action
                action_label, action_id, action_type, action_pose = \
                    coach_service.run_step(state, dqn_solver)
                coach_service.run_next_step(action_type, action_pose)
                # next state
                state_next, state_next_raw = coach_service.get_next_observation()

                # post round
                coach_service.run_post_round()

                if self.VIEW:
                    view.update()

                terminal = coach_service.is_terminal() or step > DQNSolver.MAX_EPISODE
                reward = coach_service.get_reward(step, action_id, terminal)
                dqn_solver.remember(state, action_id, reward, state_next, terminal)
                if terminal:
                    print(
                        "\nRun: " + str(coach_run) +
                        ", exploration: " + str(round(dqn_solver.exploration_rate, 3)) +
                        ", step: " + str(step))
                    if len(self.score_logger.scores) > 0 and \
                            step < min(self.score_logger.scores):
                        dqn_solver.save_model(step)
                        dqn_solver.next_run(0.9)
                    else:
                        dqn_solver.next_run(0.998)
                    self.score_logger.add_score(step, coach_run)
                    break
                else:
                    dqn_solver.experience_replay()

                if self.VIEW:
                    view_data.update({f"Act-{action_label}": action_pose},
                                     {f"Reward": reward},
                                     {f"Distan": " : ".join(
                                         str(round(x, 3)) for x in
                                         state_next[0][
                                            coach_service.LH_DIST_START_AT:
                                            coach_service.LH_DIST_END_AT
                                         ])},
                                     {f"Step": step})
                    view.update(view_data)

                delay(self.GAME_DELAY)


if __name__ == '__main__':
    coach = Coach()

    # profiler = Profiler()
    # profiler.start()

    coach.train()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
