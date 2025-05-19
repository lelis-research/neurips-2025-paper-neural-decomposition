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
    mode                                         = ["train_option"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results"
    device:                   str                = torch.device("cpu")

    # ----- tune experiment settings -----
    tuning_nametag:           str                = "No_Options"
    num_trials:               int                = 200    
    steps_per_trial:          int                = 300_000
    param_ranges                                 = {
                                                        "step_size":         [3e-5, 3e-4, 3e-3],
                                                        # "num_minibatches":   [16,   128],
                                                        # "rollout_steps":     [500, 1000, 2000],
                                                        "entropy_coef":      [0, 0.01],
                                                    }
    tuning_env_name:          str                = "Medium_Maze"
    tuning_env_params                          = {"continuing_task": False, "reward_type": "sparse"}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]
    tuning_env_max_steps:     int              = 500
    tuning_seeds                               = [10000, 20000, 30000]
    exhaustive_search:        bool             = True
    # num_grid_points:          int              = 5
    option_path_tuning                         = [
                                                    # f"Options_{'Transfer'}_Maze_m_Seed_{10000}/selected_options.pt",
                                                    # f"Options_{'Transfer'}_Maze_m_Seed_{20000}/selected_options.pt",
                                                    # f"Options_{'Transfer'}_Maze_m_Seed_{30000}/selected_options.pt",
                                                ]
                                                  

    # ----- train experiment settings -----
    agent_class:              str                = "PPOAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent
    seeds                                        = list(range(1000, 11000, 1000))
    exp_total_steps:          int                = 1_000_000
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    nametag:                  str                = "base"#+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    training_env_name:        str                = "Maze_1m" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {"continuing_task": False, "reward_type": "sparse"} #{"include_cfrc_ext_in_observation":False}
    training_env_wrappers                        = default_env_wrappers(training_env_name)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = None #human, None, rgb_array_list, rgb_array
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
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    entropy_coef:             float              = float(os.environ.get("ENTROPY_COEF", 0.0))
    critic_coef:              float              = 0.5
    clip_ratio:               float              = 0.2
    flag_clip_vloss:          bool               = True
    flag_norm_adv:            bool               = True
    max_grad_norm:            float              = 0.5
    flag_anneal_var:          bool               = False
    var_coef:                 float              = 0.0

    # ----- plot setting -----
    pattern                                      = {
                                                        "No Options":                           "Medium_Maze_*_1000000_sparse_success_No_Options",
                                                        
                                                        "BasePolicy Transfer Options":          "Options_Transfer_Maze_m_Seed_*_Medium_Maze_selected_options",
                                                        
                                                        "DecWhole 5 Options":                   "Options_DecWhole_Maze_m_Seed_*_Medium_Maze_selected_options_5",
                                                        "DecWhole 10 Options":                  "Options_DecWhole_Maze_m_Seed_*_Medium_Maze_selected_options_10",
                                                        
                                                        "FineTune 5 Options":                   "Options_FineTune_Maze_m_Seed_*_Medium_Maze_selected_options_5",
                                                        "FineTune 10 Options":                  "Options_FineTune_Maze_m_Seed_*_Medium_Maze_selected_options_10",
                                                        
                                                        "Mask 5 Options":                       "Options_Mask_Maze_m_Seed_*_Medium_Maze_selected_options_5",
                                                        "Mask 10 Options":                      "Options_Mask_Maze_m_Seed_*_Medium_Maze_selected_options_10",
                                                    }
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = "Medium_Maze_Comparison"

    # ----- Option setting -----
    tmp_seed = int(os.environ.get("TMP_SEED", 1000))
    tmp_opt="FineTune"
    env_agent_list                               = [
                                                    {"env_name": "Maze_1m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_1m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_1m")[1],
                                                     "agent_path": f"Maze_1m_{tmp_seed}_1000000_base",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_2m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_2m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_2m")[1],
                                                     "agent_path": f"Maze_2m_{tmp_seed}_1000000_base",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_3m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_3m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_3m")[1],
                                                     "agent_path": f"Maze_3m_{tmp_seed}_1000000_base",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_4m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_4m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_4m")[1],
                                                     "agent_path": f"Maze_4m_{tmp_seed}_1000000_base",
                                                     "env_max_steps":500},
                                                     
                                                    ]
    option_save_results:      bool               = True
    option_exp_name:          str                = f"Options_{tmp_opt}_Maze_m_Seed_{tmp_seed}"
    max_num_options:          int                = int(os.environ.get("MAX_NUM_OPTIONS", 5))
    
    # ----- train option experiment settings -----
    sub_trajectory_min_len:   int                = 2
    sub_trajectory_max_len:   int                = 24
    mask_epochs:              int                = 300 # number of epochs to train the mask
    
    hc_iterations:            int                = 50 # hill climbing iterations
    hc_restarts:              int                = 150 # hill climbing restarts
    hc_neighbor_samples:      int                = 100 # number of neighbors to sample for hill climbing
    action_dif_tolerance:     float              = 0.4 # tolerance for action difference
    baseline:                 str                = tmp_opt #Mask, FineTune, DecWhole, Transfer
    num_worker:               int                = 32

    # ----- test option experiment settings -----
    test_option_env_name:     str                = os.environ.get("TEST_OPTION_ENV_NAME", "Large_Maze") #Medium_Maze, Large_Maze, Hard_Maze
    test_option_env_params                       = {"continuing_task": False, "reward_type": "sparse"}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name)[1]
    test_option_env_max_steps                    = 500

    
    test_option_render_mode:   str               = "rgb_array" #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = 1000

    exp_options_total_steps:   int               = 1_000_000
    exp_options_total_episodes:int               = 0




    