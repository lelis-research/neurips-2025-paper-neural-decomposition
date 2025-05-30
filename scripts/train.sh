#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G       # memory per node
#SBATCH --time=00:40:00      # time (DD-HH:MM)
#SBATCH --output=UnlockEnv/%A-%a.out  # %N for node name, %j for jobID
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=aip-lelis
#SBATCH --array=0-89


source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
#     --seed $SLURM_ARRAY_TASK_ID\
#     --env_id "MiniGrid-FourRooms-v0"\
#     --num_steps 722\
#     --game_width 9\
#     --total_timesteps 1000000\
#     --save_run_info 1\
#     --method "options"\
#     --option_mode "didec-reg"

    python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed $SLURM_ARRAY_TASK_ID\
    --env_id "MiniGrid-Unlock-v0"\
    --num_steps 128\
    --game_width 9\
    --total_timesteps 1000000\
    --save_run_info 0\
    --method "no_options"