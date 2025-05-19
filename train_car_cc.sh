#!/usr/bin/env bash
#SBATCH --job-name=car
#SBATCH --time=0-03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-lelis
#SBATCH --array=1-30
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

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

# Compute the seed for this array task
export SEED=$(( SLURM_ARRAY_TASK_ID * 1000 ))
echo "Running experiment with SEED=$SEED"

# Run your training script
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_car.py