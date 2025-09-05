#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:50:00
#SBATCH --output=combotrainwwalls/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-29

cd /home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

source /home/rezaabdz/scratch/envs/venv2/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

idx=$(( $SLURM_ARRAY_TASK_ID + 0 ))

# ComboGrid config (commented out)
# python3.11 ~/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
#     --seed "${idx}"\\
#     --env_id "ComboGrid"\\
#     --num_steps 2000\\
#     --game_width 8\\
#     --total_timesteps 1000000\\
#     --save_run_info 0\\
#     --method "no_options"\\
#     --option_mode "vanilla"\\
#     --mask_type "both"\\
#     --sweep_run 1\\
#     --update_epochs 10

# MiniHack (Corridor-R2) config
python3.11 -m pipelines.train_ppo \
    --seed "${idx}"\
    --env_id "MiniHack-Corridor-R2-v0"\
    --num_steps 2000\
    --game_width 9\
    --total_timesteps 1000000\
    --save_run_info 0\
    --method "no_options"\
    --option_mode "vanilla"\
    --mask_type "both"\
    --sweep_run 1\
    --update_epochs 10

    # python3.11 ~/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    # --seed $SLURM_ARRAY_TASK_ID\
    # --env_id "MiniGrid-Unlock-v0"\
    # --num_steps 2000\
    # --game_width 9\
    # --total_timesteps 1000000\
    # --save_run_info 0\
    # --method "no_options"
