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
        env_wrappers= ["ViewSize", "FlattenOnehotObj", "FixedSeed"]
        wrapping_params = [{"agent_view_size": 5}, {}, {"seed": kwargs["env_seed"]}]

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
    mode                                         = ["plot"] # train, test, plot, tune, train_option, test_option
    res_dir:                  str                = "Results_MiniGrid_A2C_ReLU"
    device:                   str                = torch.device("cpu")

    # ----- train experiment settings -----
    agent_class:              str                = "A2CAgent" # PPOAgent, ElitePPOAgent, RandomAgent, SACAgent, DDPGAgent, A2CAgent
    seeds                                        = [int(os.environ.get("SEED", 1000))] 
    exp_total_steps:          int                = 2_500_000 
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    env_seed:                 int                = int(os.environ.get("ENV_SEED", 19000))
    nametag:                  str                = os.environ.get("NAMETAG", "") # +datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_workers:              int                = 1 # Number of parallel workers for training

    training_env_name:        str                = "MiniGrid-FourRooms-v0" # Medium_Maze, Large_Maze, Hard_Maze
    training_env_params                          = {} 
    training_env_wrappers                        = default_env_wrappers(training_env_name, env_seed=env_seed)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name, env_seed=env_seed)[1]
    training_env_max_steps:   int                = 500
    training_render_mode:     str                = "rgb_array" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = None
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
    

    # ----- plot setting -----
    num_distractors:           int                = int(os.environ.get("NUM_DISTRACTORS", 50)) # Number of distractors in the environment
    pattern                                      = {
                                                        # *********************   No Distractors   *********************
                                                        # "No Options_1":  "MiniGrid-FourRooms-v0_*_3000000_stepsize_0.01",
                                                        # "No Options_2":  "MiniGrid-FourRooms-v0_*_3000000_stepsize_0.001",
                                                        # "No Options_3":  "MiniGrid-FourRooms-v0_*_3000000_stepsize_0.0001", #best
                                                        
                                                        # "Transfer_1":       "Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_stepsize_0.01", 
                                                        # "Transfer_2":       "Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_stepsize_0.001",
                                                        # "Transfer_3":       "Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_stepsize_0.0001", #best
                                                        
                                                        # "DecWhole5_1":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "DecWhole5_2":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001", 
                                                        # "DecWhole5_3":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "DecWhole10_1":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "DecWhole10_2":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001", 
                                                        # "DecWhole10_3":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best
                                                        
                                                        # "DecWhole20_1":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.01",
                                                        # "DecWhole20_2":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.001", 
                                                        # "DecWhole20_3":       "Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.0001", #best
                                                        
                                                        # "FineTune5_1":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "FineTune5_2":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        # "FineTune5_3":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "FineTune10_1":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "FineTune10_2":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001", 
                                                        # "FineTune10_3":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best 
                                                        
                                                        # "FineTune20_1":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.01",
                                                        # "FineTune20_2":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.001", 
                                                        # "FineTune20_3":      "Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.0001", #best 
                                                        
                                                        # "MaskNetwork5_1":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "MaskNetwork5_2":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        # "MaskNetwork5_3":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "MaskNetwork10_1":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "MaskNetwork10_2":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001",
                                                        # "MaskNetwork10_3":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best
                                                        
                                                        # "MaskNetwork20_1":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.01",
                                                        # "MaskNetwork20_2":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.001",
                                                        # "MaskNetwork20_3":      "Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.0001", #best
                                                        
                                                        # "MaskInput5_1":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "MaskInput5_2":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        # "MaskInput5_3":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "MaskInput10_1":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "MaskInput10_2":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001",
                                                        # "MaskInput10_3":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best
                                                        
                                                        # "MaskInput20_1":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.01",
                                                        # "MaskInput20_2":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.001",
                                                        # "MaskInput20_3":      "Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.0001", #best
                                                        
                                                        # "MaskBoth5_1":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.01",
                                                        # "MaskBoth5_2":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.001",
                                                        # "MaskBoth5_3":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_stepsize_0.0001", #best
                                                        
                                                        # "MaskBoth10_1":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.01",
                                                        # "MaskBoth10_2":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.001",
                                                        # "MaskBoth10_3":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_stepsize_0.0001", #best
                                                        
                                                        # "MaskBoth20_1":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.01",
                                                        # "MaskBoth20_2":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.001",
                                                        # "MaskBoth20_3":      "Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_stepsize_0.0001", #best
                                                        
                                                        # *********************   Distractors N  *********************
                                                        # "D_No Options_1":  "MiniGrid-FourRooms-v0_*_3000000_distractors_20_stepsize_0.01",
                                                        # "D_No Options_2":  "MiniGrid-FourRooms-v0_*_3000000_distractors_20_stepsize_0.001",
                                                        # "D_No Options_3":  "MiniGrid-FourRooms-v0_*_3000000_distractors_20_stepsize_0.0001", #best

                                                        # "D_Transfer_1":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_Transfer_2":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_Transfer_3":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_DecWhole5_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_DecWhole5_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_DecWhole5_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_DecWhole10_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_DecWhole10_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_DecWhole10_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_DecWhole20_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_DecWhole20_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_DecWhole20_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_FineTune5_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_FineTune5_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_FineTune5_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_FineTune10_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_FineTune10_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_FineTune10_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_FineTune20_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_FineTune20_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_FineTune20_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskNetwork5_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskNetwork5_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskNetwork5_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskNetwork10_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskNetwork10_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "D_MaskNetwork10_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskNetwork20_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskNetwork20_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskNetwork20_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskInput5_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskInput5_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskInput5_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskInput10_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskInput10_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskInput10_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskInput20_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskInput20_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "D_MaskInput20_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskBoth5_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskBoth5_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskBoth5_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskBoth10_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskBoth10_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "D_MaskBoth10_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001", #best

                                                        # "D_MaskBoth20_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.01",
                                                        # "D_MaskBoth20_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        # "D_MaskBoth20_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001", #best
                                                                         
                                                    }
    

    
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = f"4Rooms_{num_distractors}_Distractions_Only_masking"

    # ----- Option setting -----
    tmp_seed = int(os.environ.get("TMP_SEED", 1000))
    tmp_opt= os.environ.get("TMP_OPT", "FineTune") # Mask, FineTune, DecWhole, Transfer
    
    env_agent_list                               = [
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=1)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=1)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_1",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=2)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=2)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_2",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=3)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=3)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_3",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=4)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=4)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_4",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=5)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=5)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_5",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=6)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=6)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_6",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=7)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=7)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_7",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=8)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=8)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_8",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=9)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=9)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_9",
                                                     "env_max_steps":500},
                                                    
                                                    {"env_name": "MiniGrid-SimpleCrossingS9N1-v0", 
                                                     "env_params": {},
                                                     "env_wrappers": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=10)[0],
                                                     "env_wrapping_params": default_env_wrappers("MiniGrid-SimpleCrossingS9N1-v0", env_seed=10)[1],
                                                     "agent_path": f"MiniGrid-SimpleCrossingS9N1-v0_{tmp_seed}_1000000_base_10",
                                                     "env_max_steps":500},
                                                     
                                                    ]
    option_exp_name:          str                = f"Options_{tmp_opt}_SimpleCrossing_Seed_{tmp_seed}"
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
    
    # ----- test option experiment settings -----
    option_save_results:      bool               = True
    option_name_tag:          str                = f"stepsize_{step_size}"
    test_option_env_name:     str                = os.environ.get("TEST_OPTION_ENV_NAME", "MiniGrid-FourRooms-v0") #Medium_Maze, Large_Maze, Hard_Maze
    test_option_env_params                       = {}
    test_option_env_wrappers                     = default_env_wrappers(test_option_env_name, env_seed=19000)[0]
    test_option_wrapping_params                  = default_env_wrappers(test_option_env_name,  env_seed=19000)[1]
    test_option_env_max_steps                    = 500

    
    test_option_render_mode:   str               = "rgb_array" #human, None, rgb_array_list, rgb_array
    option_save_frame_freq:    int               = None

    exp_options_total_steps:   int               = 2_500_000
    exp_options_total_episodes:int               = 0