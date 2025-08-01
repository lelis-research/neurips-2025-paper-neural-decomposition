#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:40:00
#SBATCH --output=FourRooms-didec/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --array=0-59

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed $SLURM_ARRAY_TASK_ID\
    --env_id "MiniGrid-FourRooms-v0"\
    --num_steps 2000\
    --game_width 9\
    --total_timesteps 1000000\
    --save_run_info 1\
    --method "options"\
    --option_mode "didec"\
    --mask_type "input"\
    --sweep_run 1

    # python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    # --seed $SLURM_ARRAY_TASK_ID\
    # --env_id "MiniGrid-Unlock-v0"\
    # --num_steps 2000\
    # --game_width 9\
    # --total_timesteps 1000000\
    # --save_run_info 0\
    # --method "no_options"
