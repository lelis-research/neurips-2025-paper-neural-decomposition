#!/usr/bin/env bash
#SBATCH --job-name=minigrid
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-lelis
#SBATCH --array=1-90           # 90 tasks total, max 30 running at once
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# ————— cd into your codebase —————
cd ~/scratch/neurips-2025-paper-neural-decomposition

# ————— Load modules & env —————
module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
source /home/aghakasi/ENV/bin/activate

# ————— Pin threads —————
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# ---------------------------------------------------
# Map SLURM_ARRAY_TASK_ID (1…90) → (run_idx:1…30, stepsize_idx:0…2)
# ---------------------------------------------------
IDX=$SLURM_ARRAY_TASK_ID        # 1 … 90
run_idx=$(( (IDX - 1) % 30 + 1 ))       # 1 … 30
step_idx=$(( ( (IDX - 1) / 30 ) ))      # integer division → 0,1,2

# List your three step‐sizes here in matching order
STEPSIZES=(0.01 0.001 0.0001)
export STEP_SIZE=${STEPSIZES[$step_idx]}

# Derive your seed however you like (here: mult of 1000)
export SEED=$(( run_idx * 1000 ))
export NAMETAG="distractors_50_stepsize_${STEP_SIZE}"
echo ">>> TASK $IDX: run_idx=$run_idx, STEP_SIZE=$STEP_SIZE, SEED=$SEED"

# ————— Run your training with those env vars —————
python -u main.py --config_path configs/config_a2c_train.py