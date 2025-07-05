#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --output=selecting_options/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --array=0-29 #1080

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
    --seed $SLURM_ARRAY_TASK_ID\
    --game_width 9\
    --cpus=$SLURM_CPUS_PER_TASK
