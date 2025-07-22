import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime
import torch
import os
import random

from Environments.ComboGrid.GetEnvironment import COMBOGRID_ENV_LST

GAME_WIDTH = int(os.environ.get("GAME_WIDTH", 5))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 64))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 100_000))
SEED = int(os.environ.get("SEED", 1))
ENV_SEED = int(os.environ.get("ENV_SEED", 0))
MODE = os.environ.get("MODE", "train_option").split("-")
ENV_SEEDS = list(map(int, os.environ.get("ENV_SEEDS", "0 1 2 3").split(" ")))
ENV_NAME = os.environ.get("ENV_NAME", "ComboGrid")
RES_DIR = os.environ.get("RES_DIR", f"Results_{ENV_NAME}_gw{GAME_WIDTH}h{HIDDEN_SIZE}_A2C_ReLU")
AGENT_CLASS = os.environ.get("AGENT_CLASS", "A2CAgent")


# np.random.seed(SEED)
# torch.manual_seed(SEED)
# random.seed(SEED)

def default_env_wrappers(env_name, **kwargs):
    
    # print(f"No default wrappers for {env_name} environment!")
    env_wrappers= []
    wrapping_params = []

    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = MODE # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = RES_DIR
    device:                   str                = torch.device("cpu")
    game_width:               int                = GAME_WIDTH
    hidden_size:              int                = HIDDEN_SIZE

    critic_hidden_size:       int        = 200 # Hidden size for the critic network, used in A2CAgent       
    
    # ----- train experiment settings -----
    agent_class:              str                = AGENT_CLASS # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [SEED] 
    exp_total_steps:          int                = TOTAL_STEPS 
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    env_seed:                 int                = ENV_SEED
    nametag:                  str                = f'env_{ENV_SEED}' # +datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_workers:              int                = 1 # Number of parallel workers for training

    training_env_name:        str                = ENV_NAME # Medium_Maze, Large_Maze, Hard_Maze
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

    test_env_name:            str                = ENV_NAME
    # test_agent_path:          str                = ""
    test_agent_path:          str                = f"{test_env_name}_{SEED}_{exp_total_steps}_{nametag}"
    test_env_params                              = {"env_seed": env_seed, "step_reward": 0, "goal_reward": 10 if env_seed == 12 else 1, "game_width": GAME_WIDTH}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- tune experiment settings -----

    tuning_nametag:           str              = f"gw{GAME_WIDTH}-h{HIDDEN_SIZE}-vanilla"
    num_trials:               int              = 10   
    steps_per_trial:          int              = 100_000
    # param_ranges                               = {
    #                                             "gamma": [0.95, 0.97, 0.99],  # Typical range for discount factors
    #                                             "lamda": [0.90, 0.95, 0.97, 0.99],  # GAE lambda
    #                                             "epochs": [3, 5, 10],  # PPO epoch count per update
    #                                             "rollout_steps": [1024, 2048, 4096],  # Number of steps per rollout
    #                                             "num_minibatches": [16, 32, 64],  # Used for batch splitting
    #                                             "step_size": [
    #                                                 1e-6, 3e-6, 1e-5, 3e-5, 5e-5,
    #                                                 1e-4, 2e-4, 3e-4, 1e-3, 3e-3
    #                                             ],
    #                                             "entropy_coef": [0.0, 0.01, 0.02, 0.05],  # Encourages exploration
    #                                             "critic_coef": [0.25, 0.5, 1.0],  # Value loss weight
    #                                             "clip_ratio": [0.1, 0.2, 0.3],  # PPO clip parameter
    #                                             "max_grad_norm": [0.1, 0.5, 1.0],  # Gradient clipping
    #                                             "var_coef": [0.0, 0.01, 0.1],  # Optional value loss penalty
    #                                             "l1_lambda": [0.0, 1e-6, 1e-5, 1e-4]
    #                                                 }
    param_ranges                               = {
                                                        "step_size":[
                                                                        1e-6,
                                                                        3e-6,
                                                                        1e-5,
                                                                        3e-5,
                                                                        5e-5,
                                                                        1e-4,
                                                                        2e-4,
                                                                        3e-4,
                                                                        1e-3,
                                                                        3e-3
                                                                    ],
                                                    }
    tuning_env_name:          str              = ENV_NAME
    tuning_env_params                          = {"env_seed": ENV_SEED, "step_reward": 0, "goal_reward": 10, "game_width": GAME_WIDTH}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]
    tuning_env_max_steps:     int              = 500
    tuning_seeds                               = [0]
    exhaustive_search:        bool             = True
    num_grid_points:          int              = 10
    option_path_tuning                         = []
    tuning_storage:           str              = "sqlite:///optuna.db"
    n_trials_per_job:         int              = 10

    # ----- A2C hyper‑parameters -----
    gamma:                    float              = 1
    lamda:                    float              = 0.95
    rollout_steps:            int                = 7
    step_size:                float              = float(os.environ.get("STEP_SIZE", 3e-4))
    

    # ----- PPO hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95

    epochs:                   int                = 10
    total_steps:              int                = TOTAL_STEPS
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
    l1_lambda:                float              = float(os.environ.get("L1_LAMBDA", 1e-5))

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
                                                    {"env_name": ENV_NAME, 
                                                     "env_params": {"env_seed": ENV_SEEDS[0], "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers(ENV_NAME)[0],
                                                     "env_wrapping_params": default_env_wrappers(ENV_NAME)[1],
                                                     "agent_path": f"{ENV_NAME}_{tmp_seed}_{str(exp_total_steps)}_env_{ENV_SEEDS[0]}",
                                                     "env_max_steps":500},

                                                    {"env_name": ENV_NAME, 
                                                     "env_params": {"env_seed": ENV_SEEDS[1], "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers(ENV_NAME)[0],
                                                     "env_wrapping_params": default_env_wrappers(ENV_NAME)[1],
                                                     "agent_path": f"{ENV_NAME}_{tmp_seed}_{str(exp_total_steps)}_env_{ENV_SEEDS[1]}",
                                                     "env_max_steps":500},

                                                    {"env_name": ENV_NAME, 
                                                     "env_params": {"env_seed": ENV_SEEDS[2], "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers(ENV_NAME)[0],
                                                     "env_wrapping_params": default_env_wrappers(ENV_NAME)[1],
                                                     "agent_path": f"{ENV_NAME}_{tmp_seed}_{str(exp_total_steps)}_env_{ENV_SEEDS[2]}",
                                                     "env_max_steps":500},
                                                     
                                                    {"env_name": ENV_NAME, 
                                                     "env_params": {"env_seed": ENV_SEEDS[3], "step_reward": 0, "goal_reward": 1, "game_width": GAME_WIDTH},
                                                     "env_wrappers": default_env_wrappers(ENV_NAME)[0],
                                                     "env_wrapping_params": default_env_wrappers(ENV_NAME)[1],
                                                     "agent_path": f"{ENV_NAME}_{tmp_seed}_{str(exp_total_steps)}_env_{ENV_SEEDS[3]}",
                                                     "env_max_steps":500},
                                                    ]
    option_exp_name:          str                = f"Options_{tmp_opt}_{ENV_NAME}_Seed_{tmp_seed}_{mask_type}"
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
    test_option_env_name:     str                = os.environ.get("TEST_OPTION_ENV_NAME", ENV_NAME) #Medium_Maze, Large_Maze, Hard_Maze
    test_option_env_params                       = {"env_seed": 12, "step_reward": 0, "goal_reward": 10, "game_width": GAME_WIDTH}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name)[1]
    test_option_env_max_steps                    = 500

    
    test_option_render_mode:   str               = "" #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = None

    exp_options_total_steps:   int               = TOTAL_STEPS
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