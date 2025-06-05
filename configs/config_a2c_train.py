import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os

from Environments.MiniGrid.GetEnvironment import MINIGRID_ENV_LST, layout_1
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
                       "FixedRandomDistractor",
                       ]
        wrapping_params = [{"agent_view_size": 9}, 
                           {}, 
                           {"seed": kwargs["env_seed"]}, 
                           {"num_distractors": kwargs["env_num_distractor"]},
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
    mode                                         = ["train"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_MiniGrid_A2C_ReLU"
    device:                   str                = torch.device("cpu")

    # ----- train experiment settings -----
    agent_class:              str                = "A2CAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [int(os.environ.get("SEED", 1000))] 
    exp_total_steps:          int                = 2_000_000 
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    env_seed:                 int                = int(os.environ.get("ENV_SEED", 19000))
    nametag:                  str                = os.environ.get("NAMETAG", "") # +datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_workers:              int                = 1 # Number of parallel workers for training

    training_env_name:        str                = "MiniGrid-FourRooms-v0" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {} 
    num_distractor:           int                = int(os.environ.get("NUM_DISTRACTOR", 0))
    training_env_wrappers                        = default_env_wrappers(training_env_name, env_seed=env_seed, env_num_distractor=num_distractor)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name, env_seed=env_seed, env_num_distractor=num_distractor)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = "rgb_array" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = None
    load_agent:               str                = None # "car-test_1000_1000000_Tanh64_20250503_222014"

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    

