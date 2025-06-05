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

    

    # ----- plot setting -----
    num_distractors:           int                = int(os.environ.get("NUM_DISTRACTORS", 0)) # Number of distractors in the environment
    pattern                                      = {
                                                     
                                                        
                                                        # *********************   Distractors N  *********************
                                                        "No Options_1":  f"MiniGrid-FourRooms-v0_*_2000000_distractors_{num_distractors}_stepsize_0.001",
                                                        "No Options_2":  f"MiniGrid-FourRooms-v0_*_2000000_distractors_{num_distractors}_stepsize_0.0001",
                                                        "No Options_3":  f"MiniGrid-FourRooms-v0_*_2000000_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "Transfer_1":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.001",
                                                        "Transfer_2":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.0001",
                                                        "Transfer_3":  f"Options_Transfer_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "DecWhole5_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        "DecWhole5_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001",
                                                        "DecWhole5_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "DecWhole10_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "DecWhole10_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001",
                                                        "DecWhole10_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "DecWhole20_1":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "DecWhole20_2":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001",
                                                        "DecWhole20_3":  f"Options_DecWhole_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "FineTune5_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        "FineTune5_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001",
                                                        "FineTune5_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "FineTune10_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "FineTune10_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001",
                                                        "FineTune10_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "FineTune20_1":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "FineTune20_2":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001",
                                                        "FineTune20_3":  f"Options_FineTune_SimpleCrossing_Seed_*_None_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskNetwork5_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskNetwork5_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskNetwork5_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskNetwork10_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskNetwork10_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskNetwork10_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskNetwork20_1":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskNetwork20_2":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskNetwork20_3":  f"Options_Mask_SimpleCrossing_Seed_*_network_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskInput5_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskInput5_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskInput5_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskInput10_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskInput10_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskInput10_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskInput20_1":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskInput20_2":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskInput20_3":  f"Options_Mask_SimpleCrossing_Seed_*_input_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskBoth5_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskBoth5_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskBoth5_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_5_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskBoth10_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskBoth10_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskBoth10_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_10_distractors_{num_distractors}_stepsize_0.00001", #best

                                                        "MaskBoth20_1":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.001",
                                                        "MaskBoth20_2":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.0001",
                                                        "MaskBoth20_3":  f"Options_Mask_SimpleCrossing_Seed_*_both_MiniGrid-FourRooms-v0_selected_options_20_distractors_{num_distractors}_stepsize_0.00001", #best
                                                                         
                                                    }
    

    
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = f"4Rooms_{num_distractors}_Distractions"

