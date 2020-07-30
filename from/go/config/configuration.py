import os

from lib.utils import dotdict


class Configuration:
    # specifically choose TF cpu if needed. This will have no effect if GPU is not present
    USE_TF_CPU = True

    # helper path so model weights are imported and exported correctly when transferring project
    PATH = os.path.dirname(os.path.realpath(__file__))

    # Show initial TF configuration when TF is getting initialized
    SHOW_TENSORFLOW_GPU = False

    # Disable TF warnings
    DISABLE_TENSORFLOW_WARNING = True

    # Show initial Pygame welcome message when Pygame is getting initialized
    SHOW_PYGAME_WELCOME = False

    # If keras should output while fitting data
    VERBOSE_MODEL_FIT = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch.

    # Maximum number of fps Pygame will render game at. Only relevant when running with verbose > 3
    FPS = 5

    # ##################################
    # ########### ENCODERS #############
    # ##################################

    # Defining number of encoders
    NUM_ENCODERS = 14

    # Setting indexes to each encoder
    ISLAND_IDX = 0
    ENERGY_IDX = 1
    P_NAME_IDX = 2
    A_TYPE_IDX = 3
    PL_SCORE_W1_IDX = 4
    PL_SCORE_W2_IDX = 5
    PL_ENERGY_W1_IDX = 6
    PL_ENERGY_W2_IDX = 7
    LH_ENERGY_IDX = 8
    LH_OWNER_IDX = 9
    LH_KEY_IDX = 10
    LH_CONN_IDX = 11
    LH_TRI_IDX = 12
    TIME_IDX = 13

    # Connections and polygons
    CONN_X0_IDX = 0
    CONN_Y0_IDX = 1
    CONN_X1_IDX = 2
    CONN_Y1_IDX = 3
    TRI_X0_IDX = 0
    TRI_Y0_IDX = 1
    TRI_X1_IDX = 2
    TRI_Y1_IDX = 3
    TRI_X2_IDX = 4
    TRI_Y2_IDX = 5
    TRI_CC_IDX = 6

    # ##################################
    # ########### ACTORS ###############
    # ##################################

    # Dictionary for actors
    d_a_type = dotdict({
        'Work': 1,
        'Lighthouse': 2
    })

    # Reverse dictionary for actors
    d_type_rev = dotdict({
        1: 'Work',
        2: 'Lighthouse'
    })

    # ##################################
    # ########## ACTIONS ###############
    # ##################################

    # Dictionary for actions and which actor can execute them
    d_acts = dotdict({
        1: ["pass", "up", "down", "right", "left", "upright", "upleft", "downright", "downleft",
            "attack10", "attack30", "attack60", "attack80", "attack100",
            "connect0", "connect1", "connect2", "connect3", "connect4"],
    })

    # Reverse dictionary for actions
    d_acts_int = dotdict({
        1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # Work
    })

    # Defining all actions
    ACTS = {
        "pass": 0,

        "up": 1,
        "down": 2,
        "right": 3,
        "left": 4,
        "upright": 5,
        "upleft": 6,
        "downright": 7,
        "downleft": 8,

        "attack10": 9,
        "attack30": 10,
        "attack60": 11,
        "attack80": 12,
        "attack100": 13,

        "connect0": 14,
        "connect1": 15,
        "connect2": 16,
        "connect3": 17,
        "connect4": 18
    }

    # Reverse dictionary for all actions
    ACTS_REV = {
        0: "pass",

        1: "down",
        2: "up",
        3: "right",
        4: "left",
        5: "upright",
        6: "upleft",
        7: "downright",
        8: "downleft",

        9: "attack10",
        10: "attack30",
        11: "attack60",
        12: "attack80",
        13: "attack100",

        14: "connect0",
        15: "connect1",
        16: "connect2",
        17: "connect3",
        18: "connect4"
    }

    # Count of all actions
    NUM_ACTS = len(ACTS)

    # ####################################################################################
    # ################################## PLAYING #########################################
    # ####################################################################################

    # User shortcuts that player can use using Pygame
    d_user_shortcuts = dotdict({
        ' ': 0,  # idle
        's': 1,  # down
        'w': 2,  # up
        'd': 3,  # right
        'a': 4,  # left
        'e': 7,  # upright
        'q': 8,  # upleft
        'c': 5,  # downright
        'z': 6,  # downleft
        '1': 9,  # attack10
        '2': 10,  # attack30
        '3': 11,  # attack60
        '4': 12,  # attack80
        '5': 13,  # attack100
        'h': 14,  # connect0
        'j': 15,  # connect1
        'k': 16,  # connect2
        'l': 17,  # connect3
        'ñ': 18,  # connect4
    })

    # Reverse dictionary for user shortcuts
    d_user_shortcuts_rev = dotdict({
        0: ' ',  # idle
        1: 's',  # down
        2: 'w',  # up
        3: 'd',  # right
        4: 'a',  # left
        5: 'e',  # upright
        6: 'q',  # upleft
        7: 'c',  # downright
        8: 'z',  # downleft
        9: '1',  # attack10
        10: '2',  # attack30
        11: '3',  # attack60
        12: '4',  # attack80
        13: '5',  # attack100
        14: 'h',  # connect0
        15: 'j',  # connect1
        16: 'k',  # connect2
        17: 'l',  # connect3
        18: 'ñ',  # connect4
    })

    # Colors of actors displayed in Pygame
    d_a_color = dotdict({
        1: (230, 0, 50),  # Lighthouse : red
        2: (0, 165, 208),  # Work : blue
    })
