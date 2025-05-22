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
    mode                                         = ["test"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_car_all_action_ppo_best"
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
    agent_class:              str                = "PPOAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, DQNAgent
    seeds                                        = list(range(1000, 11000, 1000)) #[int(os.environ.get("SEED", 1000))] #
    exp_total_steps:          int                = 2_000_000
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    nametag:                  str                = os.environ.get("NAMETAG", "")

    training_env_name:        str                = "car-train" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {}#{"continuing_task": False, "reward_type": "sparse"} #{"include_cfrc_ext_in_observation":False}
    training_env_wrappers                        = default_env_wrappers(training_env_name)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = None #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = 2000
    load_agent:               str                = None #f"car-train_{seeds[0]}_1000000_"

    # ----- test experiment settings -----
    test_agent_path:          str                = os.environ.get("TEST_AGENT_PATH",  "car-train_43000_1000000_")
    test_episodes:            int                = 100
    test_seed:                int                = 0 
    save_test:                bool               = False

    test_env_name:            str                = os.environ.get("TEST_ENV_NAME",  "car-test")
    test_env_params                              = {}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- PPO hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95

    epochs:                   int                = 10
    total_steps:              int                = 5_000_000
    rollout_steps:            int                = 2750
    num_minibatches:          int                = 100
    
    flag_anneal_step_size:    bool               = True
    step_size:                float              = float(os.environ.get("STEP_SIZE", 9.5e-5))
    entropy_coef:             float              = float(os.environ.get("ENTROPY_COEF", 0.0))
    critic_coef:              float              = 0.5
    clip_ratio:               float              = 0.2
    flag_clip_vloss:          bool               = True
    flag_norm_adv:            bool               = True
    max_grad_norm:            float              = 0.5
    flag_anneal_var:          bool               = False
    var_coef:                 float              = 0.0
    l1_lambda:                float              = float(os.environ.get("L1_LAMBDA", 1e-5))
    
    # ----- DQN hyper‑parameters -----
    # gamma:                    float              = 0.99
    # step_size:                float              = float(os.environ.get("STEP_SIZE", 0.0001))
    # batch_size:               float              = int(os.environ.get("BATCH_SIZE", 64))
    # target_update_freq:       float              = int(os.environ.get("TARGET_UPDATE_FREQ", 1000))
    # epsilon:                  float              = float(os.environ.get("EPSILON", 0.01))
    # replay_buffer_cap:        float              = int(os.environ.get("REPLAY_BUFFER_CAP", 2_000_000))
    # action_res:               float              = int(os.environ.get("ACTION_RES", 3))
    
    # ----- DDPG hyper‑parameters -----
    # gamma:                    float              = 0.99
    # tau:                      float              = float(os.environ.get("TAU", 0.005))
    
    # actor_lr:                 float              = float(os.environ.get("ACTOR_LR", 0.001))
    # critic_lr:               float               = float(os.environ.get("CRITIC_LR", 0.001))
    
    # buf_size:                 float              = int(os.environ.get("BUF_SIZE", 100000))
    # batch_size:               float              = int(os.environ.get("BATCH_SIZE", 64))
    
    # noise_phi:                float              = float(os.environ.get("NOISE_PHI", 0.2))
    # ou_theta:                 float              = float(os.environ.get("OU_THETA", 0.15))
    # ou_sigma:                 float              = float(os.environ.get("OU_SIGMA", 0.2))
    
    # epsilon_end:              float              = float(os.environ.get("EPSILON_END", 0.01))
    # decay_steps:              float              = float(os.environ.get("DECAY_STEPS", 200000))

    
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
    tmp_seed = int(os.environ.get("TMP_SEED", 90000))
    tmp_opt=os.environ.get("TMP_OPT", "Mask")
    env_agent_list                               = [
                                                    {"env_name": "Maze_1m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_1m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_1m")[1],
                                                     "agent_path": f"Maze_1m_{tmp_seed}_300000_sparse_success",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_2m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_2m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_2m")[1],
                                                     "agent_path": f"Maze_2m_{tmp_seed}_300000_sparse_success",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_3m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_3m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_3m")[1],
                                                     "agent_path": f"Maze_3m_{tmp_seed}_300000_sparse_success",
                                                     "env_max_steps":500},

                                                     {"env_name": "Maze_4m", 
                                                     "env_params": {"continuing_task": False, "reward_type": "sparse"},
                                                     "env_wrappers": default_env_wrappers("Maze_4m")[0],
                                                     "env_wrapping_params": default_env_wrappers("Maze_4m")[1],
                                                     "agent_path": f"Maze_4m_{tmp_seed}_300000_sparse_success",
                                                     "env_max_steps":500},
                                                     
                                                    ]
    option_save_results:      bool               = True
    option_exp_name:          str                = f"Options_{tmp_opt}_Maze_m_Seed_{tmp_seed}"
    max_num_options:          int                = int(os.environ.get("MAX_NUM_OPTIONS", 5))
    
    # ----- train option experiment settings -----
    sub_trajectory_min_len:   int                = 2
    sub_trajectory_max_len:   int                = 24
    mask_epochs:              int                = 300 # number of epochs to train the mask
    
    hc_iterations:            int                = 200 # hill climbing iterations
    hc_restarts:              int                = 500 # hill climbing restarts
    hc_neighbor_samples:      int                = 100 # number of neighbors to sample for hill climbing
    action_dif_tolerance:     float              = 0.4 # tolerance for action difference
    baseline:                 str                = tmp_opt #Mask, FineTune, DecWhole, Transfer
    num_worker:               int                = 16

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




    