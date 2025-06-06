import os
from dataclasses import dataclass

from configs.combogrid.config_a2c_train import arguments as parent_arguments, default_env_wrappers
from configs.combogrid.config_a2c_train import GAME_WIDTH, HIDDEN_SIZE, TOTAL_STEPS, SEED, ENV_SEED

MODE = os.environ.get("MODE", "train_option").split("-")
TMP_OPT = os.environ.get("TMP_OPT", "Mask")
MASK_TYPE = None if TMP_OPT != "Mask" else os.environ.get("MASK_TYPE", "network")


@dataclass
class arguments(parent_arguments):
    
    mode = MODE # train, test, plot, tune, train_option, test_option, search_option
    agent_class: str = "A2CAgentOption"

    # ----- search option experiment settings -----
    selection_type:           str              = "local_search" # local_search, greedy 

    # ----- tune experiment settings -----

    tuning_nametag:           str              = f"{TMP_OPT}_{MASK_TYPE}" # Mask_Network, Mask_Input, Mask_Both
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
    tuning_seeds                               = []
    exhaustive_search:        bool             = True
    # num_grid_points:          int              = 5
    option_path_tuning                         = [
                                                    # f"Options_Mask_ComboGrid_Seed_{seed}_network/selected_options_5.pt" for seed in range(3)
                                                    f"Options_{TMP_OPT}_ComboGrid_Seed_{seed}_{MASK_TYPE}/selected_options_5.pt" for seed in range(3)
                                                ]
    tuning_storage:           str              = "sqlite:///optuna.db"
    n_trials_per_job:         int              = 1

    