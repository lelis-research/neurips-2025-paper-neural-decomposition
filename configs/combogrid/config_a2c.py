import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os

from Environments.ComboGrid.GetEnvironment import COMBOGRID_ENV_LST

GAME_WIDTH = 5

def default_env_wrappers(env_name, **kwargs):
    
    # print(f"No default wrappers for {env_name} environment!")
    env_wrappers= []
    wrapping_params = []

    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = ["test_option"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_ComboGrid_A2C_ReLU"
    device:                   str                = torch.device("cpu")

    # ----- train experiment settings -----
    agent_class:              str                = "A2CAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [int(os.environ.get("SEED", 1))] 
    exp_total_steps:          int                = 100_000 
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    env_seed:                 int                = int(os.environ.get("ENV_SEED", 0))
    nametag:                  str                = f'env_{os.environ.get("ENV_SEED", 0)}' # +datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_workers:              int                = 1 # Number of parallel workers for training

    training_env_name:        str                = "ComboGrid" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {"env_seed": env_seed, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH} 
    training_env_wrappers                        = default_env_wrappers(training_env_name, env_seed=env_seed)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name, env_seed=env_seed)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = "" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = None
    load_agent:               str                = None # "car-test_1000_1000000_Tanh64_20250503_222014"

    # ----- test experiment settings -----
    
    

    test_episodes:            int                = 10
    test_seed:                int                = 0 
    save_test:                bool               = False

    test_env_name:            str                = "ComboGrid"
    # test_agent_path:          str                = ""
    test_agent_path:          str                = f"{test_env_name}_{os.environ.get('SEED', 0)}_{exp_total_steps}_{nametag}"
    test_env_params                              = {"env_seed": env_seed, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    

    # ----- plot setting -----
    pattern                                      = {
                                                        # "No Options_1":  "MiniGrid-FourRooms-v0_*_1000000_stepsize_0.01",
                                                        # "No Options_2":  "MiniGrid-FourRooms-v0_*_1000000_stepsize_0.001",
                                                        # "No Options_3":  "MiniGrid-FourRooms-v0_*_1000000_stepsize_0.0001", #best
                                                        
                                                        "Transfer_1":       "Options_Transfer_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_stepsize_0.01", 
                                                        "Transfer_2":       "Options_Transfer_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_stepsize_0.001",
                                                        "Transfer_3":       "Options_Transfer_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_stepsize_0.0001", #best
                                                        
                                                        # "DecWhole5_1":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "DecWhole5_2":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001", 
                                                        "DecWhole5_3":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "DecWhole10_1":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        "DecWhole10_2":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001", #best
                                                        # "DecWhole10_3":       "Options_DecWhole_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001",
                                                        
                                                        # "FineTune5_1":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "FineTune5_2":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        "FineTune5_3":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "FineTune10_1":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        "FineTune10_2":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001", #best
                                                        # "FineTune10_3":      "Options_FineTune_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", 
                                                        
                                                        # "Mask5_1":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "Mask5_2":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        "Mask5_3":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "Mask10_1":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "Mask10_2":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001", #best
                                                        "Mask10_3":      "Options_Mask_SimpleCrossing_Seed_*_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best
                                                                         
                                                    }
    
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = "4Rooms"

    # ----- Option setting -----
    tmp_seed = int(os.environ.get("SEED", 0))
    tmp_opt= os.environ.get("TMP_OPT", "Mask") # Mask, FineTune, DecWhole, Transfer
    mask_type:                str                = None if tmp_opt != "Mask" else os.environ.get("MASK_TYPE", "network") # network, input, both
    
    env_agent_list                               = [
                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 0, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_100000_env_0",
                                                     "env_max_steps":500},

                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 1, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_100000_env_1",
                                                     "env_max_steps":500},

                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 2, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_100000_env_2",
                                                     "env_max_steps":500},
                                                     
                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 3, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_100000_env_3",
                                                     "env_max_steps":500},
                                                    
                                                     
                                                    ]
    option_exp_name:          str                = f"Options_{tmp_opt}_ComboGrid_Seed_{tmp_seed}_{mask_type}"
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

    num_worker:               int                = 8
    
    # ----- test option experiment settings -----
    option_save_results:      bool               = True
    option_name_tag:          str                = f"distractors_50_stepsize_{step_size}"
    test_option_env_name:     str                = os.environ.get("TEST_OPTION_ENV_NAME", "ComboGrid") #Medium_Maze, Large_Maze, Hard_Maze
    test_option_env_params                       = {"env_seed": 12, "step_reward": 0, "goal_reward": 10, "game_width": GAME_WIDTH}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name)[1]
    test_option_env_max_steps                    = 500

    
    test_option_render_mode:   str               = "" #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = None

    exp_options_total_steps:   int               = 1_000_000
    exp_options_total_episodes:int               = 0