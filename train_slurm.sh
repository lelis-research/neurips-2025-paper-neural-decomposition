#!/usr/bin/env bash
#SBATCH --job-name=minigrid
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-lelis
#SBATCH --array=1-30
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

cd ~/scratch/neurips-2025-paper-neural-decomposition

# ensure logs folder exists
mkdir -p logs

# ————— Load Compute Canada modules —————
module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

# ————— Activate your Python env —————
source /home/aghakasi/ENV/bin/activate

export PYTHONUNBUFFERED=1  
export FLEXIBLAS=imkl

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Compute the seed for this array task
export SEED=$(( SLURM_ARRAY_TASK_ID * 1000 ))
echo "Running experiment with SEED=$SEED"


# Run your training script
python main.py --config_path configs/config_a2c.py