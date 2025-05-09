import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch

from Environments.MiniGrid.GetEnvironment import MINIGRID_ENV_LST
from Environments.MuJoCu.GetEnvironment import MUJOCO_ENV_LST
from Environments.Car.GetEnvironment import CAR_ENV_LST

def default_env_wrappers(env_name):
    if env_name in MUJOCO_ENV_LST:
        env_wrappers= [ 
            "SuccessBonus",
            "CombineGoals", 
            
            "ClipAction", 
            "RecordReward",
            "StepReward",

            "NormalizeReward",
            "ClipReward",
            ]
        wrapping_params = [
            {"bonus":5.0}, 
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
        env_wrappers= ["ViewSize", "FlattenOnehotObj", "StepReward"]
        wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}]

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
    res_dir:                  str                = "Results"
    device:                   str                = torch.device("cpu")

    # ----- tune experiment settings -----
    num_trials:               int                = 200    
    steps_per_trial:          int                = 1_000_000
    param_ranges                                 = {
                                                        "step_size":         (1e-5, 1e-3),
                                                        "num_minibatches":   (16,   128),
                                                        "rollout_steps":     (100,   10000),
                                                        "epochs":            (5, 20),
                                                        
                                                        "clip_ratio":        (0.0, 0.5),
                                                        "entropy_coef":      (0, 0.1),
                                                        "critic_coef":       (0.3, 0.7),
                                                    }
    tuning_env_name:          str                = "car-train"
    tuning_env_params                          = {} #{"continuing_task": False}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]
    tuning_seeds                               = [1000, 2000, 3000, 4000, 5000]
    exhaustive_search:        bool              = True
    num_grid_points:          int               = 5

    # ----- train experiment settings -----
    agent_class:              str                = "PPOAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent
    seeds                                        = [10000, 20000, 30000, 40000, 50000]
    exp_total_steps:          int                = 1_000_000
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    nametag:                  str                = "sparse_success_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    training_env_name:        str                = "Maze_1m"
    training_env_params                          = {"continuing_task": False, "reward_type": "sparse"} #{"include_cfrc_ext_in_observation":False}
    training_env_wrappers                        = default_env_wrappers(training_env_name)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = "rgb_array" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = 1000
    load_agent:               str                = None # "car-test_1000_1000000_Tanh64_20250503_222014"

    # ----- test experiment settings -----
    test_agent_path:          str                = "AntMaze_R_1000_1000000_sparse_success_20250509_145731"
    test_episodes:            int                = 10
    test_seed:                int                = 0 
    save_test:                bool               = False

    test_env_name:            str                = "AntMaze_R"
    test_env_params                              = {"continuing_task": False, "reward_type": "sparse"}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- PPO hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95

    epochs:                   int                = 10
    total_steps:              int                = 1_000_000
    rollout_steps:            int                = 2048
    num_minibatches:          int                = 32
    
    flag_anneal_step_size:    bool               = True
    step_size:                float              = 3e-4
    entropy_coef:             float              = 0.0
    critic_coef:              float              = 0.5
    clip_ratio:               float              = 0.2
    flag_clip_vloss:          bool               = True
    flag_norm_adv:            bool               = True
    max_grad_norm:            float              = 0.5
    flag_anneal_var:          bool               = False
    var_coef:                 float              = 0.0

    # ----- plot setting -----
    pattern:                  str                = "Maze_1_Sparse_*_200000_*"
    smoothing_window_size:    int                = 5
    interpolation_resolution: int                = 100_000

    # ----- Option setting -----
    env_agent_list                               = [
                                                    {"env_name": "Maze_D", 
                                                     "env_params": {"continuing_task": False},
                                                     "env_wrappers": default_env_wrappers("Maze_D")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_D")[1],
                                                     "agent_path": "Maze_D_1000_30000_Tanh64_20250503_222014"},

                                                     {"env_name": "Maze_L", 
                                                     "env_params": {"continuing_task": False},
                                                     "env_wrappers": default_env_wrappers("Maze_L")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_L")[1],
                                                     "agent_path": "Maze_L_1000_30000_Tanh64_20250503_221901"},

                                                     {"env_name": "Maze_R", 
                                                     "env_params": {"continuing_task": False},
                                                     "env_wrappers": default_env_wrappers("Maze_R")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_R")[1],
                                                     "agent_path": "Maze_R_1000_30000_Tanh64_20250503_221923"},

                                                     {"env_name": "Maze_U", 
                                                     "env_params": {"continuing_task": False},
                                                     "env_wrappers": default_env_wrappers("Maze_U")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_U")[1],
                                                     "agent_path": "Maze_U_1000_30000_Tanh64_20250503_221947"},
                                                     
                                                    #  {"env_name": "Ant-v5", 
                                                    #  "env_params": {},
                                                    #  "env_wrappers": default_env_wrappers("Ant-v5")[0],
                                                    #  "env_wrapping_params": default_env_wrappers("Ant-v5")[1],
                                                    #  "agent_path": "Ant-v5_1000_1000000__20250509_104255"},
                                                     
                                                    #  {"env_name": "Ant-v5", 
                                                    #  "env_params": {},
                                                    #  "env_wrappers": default_env_wrappers("Ant-v5")[0],
                                                    #  "env_wrapping_params": default_env_wrappers("Ant-v5")[1],
                                                    #  "agent_path": "Ant-v5_1000_1000000__20250509_113113"},
                                                     
                                                     
                                                    ]
    option_save_results:      bool               = True
    option_exp_name:          str                = "test_DLRU"
    
    # ----- option experiment settings -----
    sub_trajectory_min_len:   int                = 2
    sub_trajectory_max_len:   int                = 24
    mask_epochs:              int                = 300 # number of epochs to train the mask
    
    hc_iterations:            int                = 200 # hill climbing iterations
    hc_restarts:              int                = 20 # hill climbing restarts
    hc_neighbor_samples:      int                = 50 # number of neighbors to sample for hill climbing
    action_dif_tolerance:     float              = 0.2 # tolerance for action difference

    # ----- test option experiment settings -----
    test_option_env_name:     str                = "Medium_Maze_Sparse"
    test_option_env_params                       = {"continuing_task": False}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name)[1]
    
    test_option_render_mode:   str               = None #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = 10

    exp_options_total_steps:   int               = 300_000
    exp_options_total_episodes:int               = 0


