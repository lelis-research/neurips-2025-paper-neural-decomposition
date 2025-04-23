import numpy as np
from dataclasses import dataclass, field
from typing import List
import datetime

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
            "SuccessBonus",
            ]
        wrapping_params = [
        {}, 
        {}, 
        {}, 
        {}, 
        {}
        ]

    elif env_name in MINIGRID_ENV_LST:
        env_wrappers= ["ViewSize", "FlattenOnehotObj", "StepReward"]
        wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}]

    elif env_name in CAR_ENV_LST:
        env_wrappers= ["RecordReward", 
                       "ClipAction", 
                    #    "NormalizeReward",
                    #    "ClipReward",
                    #    "StepReward",
                       ]
        wrapping_params = [{}, 
                           {}, 
                           {},
                           {},
                        #    {},
                           ]
    
    else:
        raise ValueError("No default wrappers for this environment!")
    return env_wrappers, wrapping_params


@dataclass
class arguments:
    # ----- experiment settings -----
    mode                                         = [ "train"] # train, test, plot, tune
    res_dir:                  str                = "Results"

    # ----- tune experiment settings -----
    num_trials:               int                = 200    
    steps_per_trial:          int                = 2_000_000
    param_ranges                                 = {
                                                        "clip_ratio":        [0.0, 0.5],
                                                        "step_size":         (1e-5, 1e-3),
                                                        "num_minibatches":   (16,   128),
                                                        "rollout_steps":     (100,   10000),
                                                    }
    tuning_env_name:        str                = "car-train"
    tuning_env_params                          = {}#{"continuing_task": False}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]


    # ----- train experiment settings -----
    seeds                                        = [1000, 2000]
    exp_total_steps:          int                = 2_000_000
    exp_total_episodes:       int                = 0
    save_results:             bool               = True
    nametag:                  str                = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    training_env_name:        str                = "car-train"
    training_env_params                          = {}#{"continuing_task": False}
    training_env_wrappers                        = default_env_wrappers(training_env_name)[0]
    training_wrapping_params                     = default_env_wrappers(training_env_name)[1]
    training_render_mode:     str                = "rgb_array" #human, None, rgb_array_list, rgb_array
    save_frame_freq:          int                = 200
    
    # ----- test experiment settings -----
    test_agent_path:          str                = "car-train_200_200000_20250421_184957"
    test_episodes:            int                = 3
    test_seed:                int                = 0 
    save_test:                bool               = False

    test_env_name:            str                = "car-train"
    test_env_params                              = {} #{"continuing_task": False}
    test_env_wrappers                            = default_env_wrappers(test_env_name)[0]
    test_wrapping_params                         = default_env_wrappers(test_env_name)[1]

    # ----- PPO hyper‑parameters -----
    gamma:                    float              = 0.99
    lamda:                    float              = 0.95

    epochs:                   int                = 10
    total_steps:              int                = 2_000_000
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

    # ----- plot setting -----
    pattern:                  str                = "Maze_1_Dense_*_500000_*"
    smoothing_window_size:    int                = 5
    interpolation_resolution: int                = 100_000

