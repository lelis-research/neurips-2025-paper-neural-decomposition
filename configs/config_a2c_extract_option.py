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
                    #    "FixedRandomDistractor"
                       ]
        wrapping_params = [{"agent_view_size": 9}, 
                           {}, 
                           {"seed": kwargs["env_seed"]}, 
                        #    {}
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
    mode                                         = ["train_option"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_MiniGrid_A2C_ReLU"
    device:                   str                = torch.device("cpu")




    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    



    # ----- Option setting -----
    tmp_seed = int(os.environ.get("TMP_SEED", 1000))
    tmp_opt= os.environ.get("TMP_OPT", "Transfer") # Mask, FineTune, DecWhole, Transfer
    mask_type:                str                = None if tmp_opt != "Mask" else os.environ.get("MASK_TYPE", "input") # network, input, both
    mask_reg:                 str                = None if tmp_opt != "Mask" else bool(int(os.environ.get("MASK_REG", False))) # True, False
    
    env_agent_list                               = [
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=1)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=1)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_1",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=2)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=2)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_2",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=3)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=3)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_3",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=4)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=4)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_4",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=5)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=5)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_5",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=6)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=6)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_6",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=7)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=7)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_7",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=8)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=8)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_8",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=9)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=9)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_9",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=10)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=10)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_env_10",
                                                     "env_max_steps":500},
                                                     
                                                    ]
    option_exp_name:          str                = f"Options_{tmp_opt}_SimpleCrossing_Seed_{tmp_seed}_{mask_type}_{mask_reg}" 
    max_num_options                              = None if tmp_opt == "Transfer" else int(os.environ.get("MAX_NUM_OPTIONS", 5))
    
    # ----- train option experiment settings -----
    sub_trajectory_min_len:   int                = 2
    sub_trajectory_max_len:   int                = 24
    mask_epochs:              int                = 300 # number of epochs to train the mask
    
    hc_iterations:            int                = 50 # hill climbing iterations
    hc_restarts:              int                = 150 # hill climbing restarts
    hc_neighbor_samples:      int                = 100 # number of neighbors to sample for hill climbing
    action_dif_tolerance:     float              = 0.01 # tolerance for action difference
    baseline:                 str                = tmp_opt #Mask, FineTune, DecWhole, Transfer

    num_worker:               int                = 16
    
