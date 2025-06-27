

configs = [
    {
        "PlotName": "ComboGrid", "baselines" : {
            "DecOption": "Options_DecOption_ComboGrid_Seed_0_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_network_5": "Options_Mask_ComboGrid_Seed_*_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "Vanilla": "ComboGrid_*_300000_env_12",
        },
        "GAME_WIDTH": 5, "HIDDEN_SIZE": 6,
        "minutes": 20
    },

    {
        "PlotName": "ComboGrid", "baselines" : {
            "DecOption": "Options_DecOption_ComboGrid_Seed_0_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_network_5": "Options_Mask_ComboGrid_Seed_*_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "Vanilla": "ComboGrid_*_300000_env_12",
        },
        "GAME_WIDTH": 6, "HIDDEN_SIZE": 6,
        "minutes": 20
    },

    {
        "PlotName": "ComboGrid", "baselines" : {
            "DecOption": "Options_DecOption_ComboGrid_Seed_0_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_network_5": "Options_Mask_ComboGrid_Seed_*_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "Vanilla": "ComboGrid_*_500000_env_12",
            "FineTune_5":"Options_FineTune_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_both_5":"Options_Mask_ComboGrid_Seed_*_both_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_input_5":"Options_Mask_ComboGrid_Seed_*_input_ComboGrid_selected_options_5_distractors_50_stepsize_0.0003",
            "Transfer":"Options_Transfer_ComboGrid_Seed_*_None_ComboGrid_selected_options_distractors_50_stepsize_0.003",
            "DecWhole_5": "Options_DecWhole_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
                                                        
        },
        "GAME_WIDTH": 5, "HIDDEN_SIZE": 64,
        "minutes": 20
    },
    {
        "PlotName": "ComboGrid", "baselines" : {
            "DecOption": "Options_DecOption_ComboGrid_Seed_0_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_network_5": "Options_Mask_ComboGrid_Seed_*_network_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "Vanilla": "ComboGrid_*_500000_env_12",
            "FineTune_5":"Options_FineTune_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_both_5":"Options_Mask_ComboGrid_Seed_*_both_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
            "LearnMasks_input_5":"Options_Mask_ComboGrid_Seed_*_input_ComboGrid_selected_options_5_distractors_50_stepsize_0.0003",
            "Transfer":"Options_Transfer_ComboGrid_Seed_*_None_ComboGrid_selected_options_distractors_50_stepsize_0.003",
            "DecWhole_5": "Options_DecWhole_ComboGrid_Seed_*_None_ComboGrid_selected_options_5_distractors_50_stepsize_0.003",
        },
        "GAME_WIDTH": 6, "HIDDEN_SIZE": 64,
        "minutes": 20
    },
]


import subprocess


import os
import random
import string
from pathlib import Path
import subprocess


original_cwd = os.getcwd()

for cfg in configs:
    # Base directory
    plot_full_name = f"{cfg['PlotName']}{cfg['GAME_WIDTH']}x{cfg['GAME_WIDTH']}h{cfg['HIDDEN_SIZE']}"
    base_dir = f"./{plot_full_name}/"
    base_dir = Path(base_dir)

    # Find next available v{idx} directory
    idx = 0
    while (base_dir / f"plot/v{idx}").exists():
        idx += 1
    final_dir = base_dir / f"plot/v{idx}"
    final_dir.mkdir(parents=True)

    # Change working directory
    os.chdir(final_dir)

    script = f"""#!/usr/bin/env bash
#SBATCH --job-name=plot_{plot_full_name}
#SBATCH --time=0-00:{cfg['minutes']}:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --account=def-lelis
#SBATCH --array=0
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# Move into repo
cd /home/rezaabdz/projects/def-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

# Load modules & env
module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
source /home/rezaabdz/scratch/envs/venv/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index

export SEED=$SLURM_ARRAY_TASK_ID


export GAME_WIDTH={cfg['GAME_WIDTH']}
export HIDDEN_SIZE={cfg['HIDDEN_SIZE']}
export IMAGE_BASE_DIR="{os.path.join(original_cwd, final_dir)}"

baselines=({" ".join([f"{key} {value}" for key, value in cfg['baselines'].items()])})
export BASELINES="${{baselines[@]}}"


# Run your script (it should read both $SEED and $ENV_SEED from os.environ)
python -u main.py --config_path configs/combogrid/config_a2c_plot.py"""
    

    with open("job.slurm", "w") as f:
        f.write(script)

    # Optional: Run your command here (e.g., echo test)
    print(f"Running command in {final_dir.resolve()}")
    subprocess.run(["sbatch", "job.slurm"])

    # Change directory back to original to continue loop
    os.chdir(original_cwd)  # Adjust based on depth