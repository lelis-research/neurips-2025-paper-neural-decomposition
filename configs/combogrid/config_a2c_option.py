import os
from dataclasses import dataclass

from configs.combogrid.config_a2c_train import arguments as parent_arguments, default_env_wrappers


GAME_WIDTH = int(os.environ.get("GAME_WIDTH", 5))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 64))
ENV_SEED = int(os.environ.get("ENV_SEED", 0))
MODE = os.environ.get("MODE", "train_option").split("-")
ENV_SEEDS = list(map(int, os.environ.get("ENV_SEEDS", "0 1 2 3").split(" ")))
TMP_OPT = os.environ.get("TMP_OPT", "Mask")
MASK_TYPE = None if TMP_OPT != "Mask" else os.environ.get("MASK_TYPE", "network")
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", f"./")
ENV_NAME = os.environ.get("ENV_NAME", "ComboGrid")
AGENT_CLASS = os.environ.get("AGENT_CLASS", "A2CAgentOption")

@dataclass
class arguments(parent_arguments):
    
    mode = MODE # train, test, plot, tune, train_option, test_option, search_option, analyze_option
    agent_class: str = AGENT_CLASS

    # ----- search option experiment settings -----
    selection_type:           str              = "greedy" # local_search, greedy 

    # ----- analyze experiment settings -----
    analyze_output_path: str              = OUTPUT_BASE_DIR
    analyze_input_path: str              = os.path.dirname(OUTPUT_BASE_DIR)
    # analyze_input_path: str              = OUTPUT_BASE_DIR

    # ----- tune experiment settings -----

    tuning_nametag:           str              = f"baseline_{TMP_OPT}" # Mask_Network, Mask_Input, Mask_Both
    num_trials:               int              = 1
    steps_per_trial:          int              = 150_000
    param_ranges                               = {
                                                "step_size": [0.005, 0.001, 0.003, 0.0005, 0.0001, 0.00005],
                                                "entropy_coef": [0.0, 0.05, 0.1, 0.15, 0.2],  # Encourages exploration
                                                "clip_ratio": [0.1, 0.15, 0.2, 0.25, 0.3],  # PPO clip parameter
                                                # "mask_type": ["network"],
                                                "mask_type": ["network", "input", "both"],
                                                    }
    tuning_parallel_method:   str              = "job-based" # "job-based", "process-based", "thread-based"
    tuning_job_idx:         int              = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    tuning_env_name:          str              = ENV_NAME
    tuning_env_params                          = {"env_seed": ENV_SEED, "step_reward": 0, "goal_reward": 10, "game_width": GAME_WIDTH}
    tuning_env_wrappers                        = default_env_wrappers(tuning_env_name)[0]
    tuning_wrapping_params                     = default_env_wrappers(tuning_env_name)[1]
    tuning_env_max_steps:     int              = 500
    tuning_seeds                               = []
    exhaustive_search:        bool             = True
    num_grid_points:          int              = 1
    option_path_tuning                         = [
                                                    f"Options_{TMP_OPT}_{ENV_NAME}_Seed_{seed}_{MASK_TYPE}/selected_options_5.pt" if TMP_OPT != "Transfer"
                                                    else f"Options_{TMP_OPT}_{ENV_NAME}_Seed_{seed}_{MASK_TYPE}/selected_options.pt"
                                                    for seed in [0,2,3]
                                                ]
    tuning_storage:           str              = f"optuna.db"
    n_trials_per_job:         int              = 10

    