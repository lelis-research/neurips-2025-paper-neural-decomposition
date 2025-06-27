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
BASELINES = os.environ.get("BASELINES", "").split()

BASELINES = {BASELINES[i]: BASELINES[i+1] for i in range(0, len(BASELINES), 2)} if len(BASELINES) > 0 else {}
IMAGE_BASE_DIR = os.environ.get("IMAGE_BASE_DIR", f"./")

 

def default_env_wrappers(env_name, **kwargs):
    
    # print(f"No default wrappers for {env_name} environment!")
    env_wrappers= []
    wrapping_params = []

    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = "plot" 
    res_dir:                  str                = f"Results_ComboGrid_gw{GAME_WIDTH}h{HIDDEN_SIZE}_A2C_ReLU"
    device:                   str                = torch.device("cpu")
    game_width:               int                = GAME_WIDTH
    hidden_size:              int                = HIDDEN_SIZE

    
    # ----- plot setting -----
    pattern                                      = BASELINES
    
    smoothing_window_size:    int                = 1000
    interpolation_resolution: int                = 100_000
    plot_name:                str                = "ComboGrid6*6"
    image_base_dir:         str                  = IMAGE_BASE_DIR
