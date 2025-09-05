#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=unlock-sweep/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --array=1-29 #1080

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

OMP_NUM_THREADS=1 python3.11 pipelines/base_policy_transferred.py --seed=$SLURM_ARRAY_TASK_ID
