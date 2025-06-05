import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os

from Environments.MiniGrid.GetEnvironment import MINIGRID_ENV_LST
from Environments.MuJoCu.GetEnvironment import MUJOCO_ENV_LST
from Environments.Car.GetEnvironment import CAR_ENV_LST

def default_env_wrappers(env_name, **kwargs):
    if env_name in MUJOCO_ENV_LST:
        env_wrappers= [ 
            "CombineGoals", 
            
            "ClipAction", 
            "RecordReward",
            "StepReward",

            "NormalizeReward",
            "ClipReward",
            ]
        wrapping_params = [
            {}, 
            
            {}, 
            {}, 
            {"step_reward": -1.0},
            
            {},
            {"func": lambda obs: np.clip(obs, -10, 10)},
        ]
        
        

    elif env_name in MINIGRID_ENV_LST:
        env_wrappers= ["ViewSize", 
                       "FlattenOnehotObj", 
                       "FixedSeed", 
                       "FixedRandomDistractor"
                       ]
        wrapping_params = [{"agent_view_size": 9}, 
                           {}, 
                           {"seed": kwargs["env_seed"]}, 
                           {"num_distractors": kwargs["env_num_distractor"]}
                           ]

    elif env_name in CAR_ENV_LST:
        env_wrappers= ["RecordReward", 
                       "ClipAction", 
                       "NormalizeReward",
                    #    "ClipReward",
                       ]
        wrapping_params = [{}, 
                           {}, 
                           {},
                           ]
    
    else:
        print(f"No default wrappers for {env_name} environment!")
        env_wrappers= []
        wrapping_params = []

    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = ["test_option"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_MiniGrid_A2C_ReLU"
    device:                   str                = torch.device("cpu")

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    


    # ----- Option setting -----
    tmp_seed = int(os.environ.get("TMP_SEED", 1000))
    tmp_opt= os.environ.get("TMP_OPT", "FineTune") # Mask, FineTune, DecWhole, Transfer
    mask_type:                str                = None if tmp_opt != "Mask" else os.environ.get("MASK_TYPE", "network") # network, input, both
    mask_reg:                 str                = None if tmp_opt != "Mask" else bool(int(os.environ.get("MASK_REG", False))) # True, False
    
    option_exp_name:          str                = f"Options_{tmp_opt}_SimpleCrossing_Seed_{tmp_seed}_{mask_type}"
    max_num_options                              = None if tmp_opt == "Transfer" else int(os.environ.get("MAX_NUM_OPTIONS", 5))
    
    
    # ----- test option experiment settings -----
    num_distractor:           int                = int(os.environ.get("NUM_DISTRACTOR", 0))
    option_save_results:      bool               = True
    option_name_tag:          str                = f"distractors_{num_distractor}_stepsize_{step_size}"
    test_option_env_name:     str                = os.environ.get("TEST_OPTION_ENV_NAME", "MiniGrid-FourRooms-v0") #Medium_Maze, Large_Maze, Hard_Maze
    test_option_env_params                       = {}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name, env_seed=19000, env_num_distractor=num_distractor)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name,  env_seed=19000, env_num_distractor=num_distractor)[1]
    test_option_env_max_steps                    = 500

    
    test_option_render_mode:   str               = "rgb_array" #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = None

    exp_options_total_steps:   int               = 1_500_000
    exp_options_total_episodes:int               = 0