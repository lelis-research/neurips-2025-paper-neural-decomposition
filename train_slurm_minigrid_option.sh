#!/usr/bin/env bash
#SBATCH --job-name=mask
#SBATCH --time=0-1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=64
#SBATCH --account=aip-lelis
#SBATCH --array=1-30     
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# Move into repo
cd ~/scratch/neurips-2025-paper-neural-decomposition

mkdir -p logs

# Load modules & env
module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
source /home/aghakasi/ENV/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…30

# Map to agent‐seed (1…30) 
export TMP_SEED=$(( IDX * 1000 )) 
export TMP_OPT="Mask" # Mask, FineTune, DecWhole, Transfer
export MAX_NUM_OPTIONS=20 # 5, 10, 20
export MASK_TYPE="both" # network, input, both

# Run your script (it should read both $SEED and $ENV_SEED from os.environ)
python -u main.py --config_path configs/config_a2c.py