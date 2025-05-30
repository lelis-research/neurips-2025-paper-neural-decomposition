#!/bin/bash
#SBATCH --cpus-per-task=30   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00      # time (DD-HH:MM)
#SBATCH --output=option_learning/%A-%a.out  # %N for node name, %j for jobID
#SBATCH --account=aip-lelis
#SBATCH --array=0-29


source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
    --seed $SLURM_ARRAY_TASK_ID\
    --game_width 9\
    --cpus=$SLURM_CPUS_PER_TASK
