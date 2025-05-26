import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os

from Environments.MiniGrid.GetEnvironment import MINIGRID_ENV_LST
from Environments.MuJoCu.GetEnvironment import MUJOCO_ENV_LST
from Environments.Car.GetEnvironment import CAR_ENV_LST

def default_env_wrappers(env_name):
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
        
        
        # env_wrappers= [
        #     "SuccessBonus",
            
        #     "CombineGoals",
        #     "ClipAction",
        #     # "NormalizeObs",
        #     "ClipObs",
        #     "RecordReward",
        #     "NormalizeReward",
        #     "ClipReward", 
            
        #     "AntReward",
        #     "StepReward",
        #     ]
    
        # wrapping_params = [
        #     {"bonus":100.0},
                        
        #     {},
        #     {}, 
        #     # {},
        #     {"func": lambda obs: np.clip(obs, -10, 10)}, 
        #     {}, 
        #     {},
        #     {"func": lambda reward: np.clip(reward, -10, 10)},
            
        #     {"ant_r_coef": 0.1},
        #     {}
        # ]
        

    elif env_name in MINIGRID_ENV_LST:
        env_wrappers= ["ViewSize", "FlattenOnehotObj", "FixedSeed"]
        wrapping_params = [{"agent_view_size": 5}, {}, {"seed": 1}]

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
    res_dir:                  str                = "Results_MiniGrid_A2C"
    device:                   str                = torch.device("cpu")

    # ----- train experiment settings -----
    agent_class:              str                = "A2CAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [int(os.environ.get("SEED", 1000))] 
    exp_total_steps:          int                = 300_000
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    nametag:                  str                = "base1"#+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    training_env_name:        str                = "MiniGrid-SimpleCrossingS9N1-v0" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {} 
    training_env_wrappers                        = default_env_wrappers(training_env_name)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = "rgb_array" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = 100
    load_agent:               str                = None # "car-test_1000_1000000_Tanh64_20250503_222014"

    # ----- test experiment settings -----
    test_agent_path:          str                = ""
    test_episodes:            int                = 10
    test_seed:                int                = 0 
    save_test:                bool               = False

    test_env_name:            str                = "AntMaze_R"
    test_env_params                              = {"continuing_task": False, "reward_type": "sparse"}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
  


    