#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:40:00
#SBATCH --output=unlock-sweep/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --array=40-69 #1080

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/algorithms/fine_tuning.py --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID
