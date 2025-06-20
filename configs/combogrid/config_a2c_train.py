import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os

from Environments.ComboGrid.GetEnvironment import COMBOGRID_ENV_LST

GAME_WIDTH = int(os.environ.get("GAME_WIDTH", 5))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 64))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 100_000))
SEED = int(os.environ.get("SEED", 1))
ENV_SEED = int(os.environ.get("ENV_SEED", 0))
MODE = os.environ.get("MODE", "train_option").split("-")

 

def default_env_wrappers(env_name, **kwargs):
    
    # print(f"No default wrappers for {env_name} environment!")
    env_wrappers= []
    wrapping_params = []

    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = MODE # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = f"Results_ComboGrid_gw{GAME_WIDTH}h{HIDDEN_SIZE}_A2C_ReLU"
    device:                   str                = torch.device("cpu")
    game_width:               int                = GAME_WIDTH
    hidden_size:              int                = HIDDEN_SIZE

    # ----- train experiment settings -----
    agent_class:              str                = "A2CAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [SEED] 
    exp_total_steps:          int                = TOTAL_STEPS 
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    env_seed:                 int                = ENV_SEED
    nametag:                  str                = f'env_{ENV_SEED}' # +datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_workers:              int                = 1 # Number of parallel workers for training

    training_env_name:        str                = "ComboGrid" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {"env_seed": env_seed, "step_reward": 0, "goal_reward": 1 if env_seed != 12 else 10, "game_width": GAME_WIDTH} 
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
    test_agent_path:          str                = f"{test_env_name}_{SEED}_{exp_total_steps}_{nametag}"
    test_env_params                              = {"env_seed": env_seed, "step_reward": 0, "goal_reward": 10 if env_seed == 12 else 1, "game_width": GAME_WIDTH}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- tune experiment settings -----

    tuning_nametag:           str              = f"gw{GAME_WIDTH}-h{HIDDEN_SIZE}-vanilla"
    num_trials:               int              = 10   
    steps_per_trial:          int              = 100_000
    param_ranges                               = {
                                                        "step_size":         [3e-5, 3e-4, 3e-3],
                                                    }
    tuning_env_name:          str              = "ComboGrid"
    tuning_env_params                          = {"env_seed": ENV_SEED, "step_reward": 0, "goal_reward": 10, "game_width": GAME_WIDTH}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]
    tuning_env_max_steps:     int              = 500
    tuning_seeds                               = [0]
    exhaustive_search:        bool             = True
    # num_grid_points:          int              = 5
    option_path_tuning                         = []
    tuning_storage:           str              = "sqlite:///optuna.db"
    n_trials_per_job:         int              = 1

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    

    # ----- plot setting -----
    pattern                                      = {
                                                        "FineTune_5":"Options_FineTune_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
                                                        "LearnMasks_both_5":"Options_Mask_ComboGrid_Seed_*_both_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
                                                        "LearnMasks_input_5":"Options_Mask_ComboGrid_Seed_*_input_ComboGrid_selected_options_5_distractors_50_stepsize_0.0003",
                                                        "LearnMasks_network_5":"Options_Mask_ComboGrid_Seed_*_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.0003",
                                                        "Transfer":"Options_Transfer_ComboGrid_Seed_*_None_ComboGrid_selected_options_distractors_50_stepsize_0.003",
                                                        "NoOptions":"ComboGrid_*_500000_env_12",
                                                        "DecWhole_5": "Options_DecWhole_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
                                                        # f"DecWhole_sd{i}": f"Options_DecWhole_ComboGrid_Seed_{i}_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003"
                                                        # for i in range(15)
                                                        # # f"FineTune_sd{i}": f"Options_FineTune_ComboGrid_Seed_{i}_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003"
                                                        # for i in range(15)
                                                        # f"LearnMasks_network_sd{i}": f"Options_Mask_ComboGrid_Seed_{i}_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.0003"
                                                        # for i in range(15)
                                                        # f"LearnMasks_both_sd{i}": f"Options_Mask_ComboGrid_Seed_{i}_both_ComboGrid_selected_options_5_distractors_50_stepsize_0.003"
                                                        # for i in range(8,15)
                                                        # f"Transfer_sd{i}": f"Options_Transfer_ComboGrid_Seed_{i}_None_ComboGrid_selected_options_distractors_50_stepsize_0.003"
                                                        # for i in range(15)

                                                        
                                                                         
                                                    }
    
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = "ComboGrid6*6"

    # ----- Option setting -----
    tmp_seed = SEED
    tmp_opt= os.environ.get("TMP_OPT", "Mask") # Mask, FineTune, DecWhole, Transfer, DecOption
    mask_type:                str                = None if tmp_opt not in ["Mask", "DecOption"] else os.environ.get("MASK_TYPE", "network") # network, input, both
    
    env_agent_list                               = [
                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 0, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_{str(exp_total_steps)}_env_0",
                                                     "env_max_steps":500},

                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 1, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_{str(exp_total_steps)}_env_1",
                                                     "env_max_steps":500},

                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 2, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_{str(exp_total_steps)}_env_2",
                                                     "env_max_steps":500},
                                                     
                                                    {"env_name": "ComboGrid", 
                                                     "env_params": {"env_seed": 3, "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers("ComboGrid")[0],
                                                     "env_wrapping_params": default_env_wrappers("ComboGrid")[1],
                                                     "agent_path": f"ComboGrid_{tmp_seed}_{str(exp_total_steps)}_env_3",
                                                     "env_max_steps":500},
                                                    
                                                     
                                                    ]
    option_exp_name:          str                = f"Options_{tmp_opt}_ComboGrid_Seed_{tmp_seed}_{mask_type}"
    max_num_options                              = None if tmp_opt == "Transfer" else int(os.environ.get("MAX_NUM_OPTIONS", 5))
    
    # ----- train option experiment settings -----
    sub_trajectory_min_len:   int                = 2
    sub_trajectory_max_len:   int                = 50
    mask_epochs:              int                = 300 # number of epochs to train the mask
    
    hc_iterations:            int                = 50 # hill climbing iterations
    hc_restarts:              int                = 150 # hill climbing restarts
    hc_neighbor_samples:      int                = 100 # number of neighbors to sample for hill climbing
    action_dif_tolerance:     float              = 0.01 # tolerance for action difference
    baseline:                 str                = tmp_opt #Mask, FineTune, DecWhole, Transfer

    num_worker:               int                = int(os.environ.get('SLURM_CPUS_PER_TASK', 32))
    
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

    exp_options_total_steps:   int               = 500_000
    exp_options_total_episodes:int               = 0



    # if SEED == 0:
    #     for i in [0,1,3]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 1:
    #     for i in [0]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 2:
    #     for i in [2,3]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 3:
    #     for i in [1]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 5:
    #     for i in [2]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 9:
    #     for i in [1]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 10:
    #     for i in [0]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 12:
    #     for i in [1]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 13:
    #     for i in [2]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"
    # if SEED == 14:
    #     for i in [1]:
    #         env_agent_list[i]['agent_path'] = f"ComboGrid_{SEED}_{200000}_env_{i}"