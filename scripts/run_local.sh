#!/bin/bash
#SBATCH --time=0:02:00
#SBATCH --output=driver.out
#SBATCH --account=aip-lelis

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/driver.py

    # python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    # --seed $SLURM_ARRAY_TASK_ID\
    # --env_id "MiniGrid-Unlock-v0"\
    # --num_steps 2000\
    # --game_width 9\
    # --total_timesteps 1000000\
    # --save_run_info 0\
    # --method "no_options"
