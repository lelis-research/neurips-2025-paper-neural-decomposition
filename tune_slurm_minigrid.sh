#!/usr/bin/env bash
#SBATCH --job-name=base
#SBATCH --time=0-3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --account=rrg-lelis
#SBATCH --array=1-360        
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# ————— cd into your codebase —————
cd ~/scratch/neurips-2025-paper-neural-decomposition

# ————— Load modules & env —————
# module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
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
# Number of tasks per (distractor,step) block = 30 seeds × 1 = 30
# But since there are 3 step‐sizes, each distractor‐setting covers 30×3 = 90 tasks
TASKS_PER_DIST=90

# Determine which distractor‐setting (0,10,20,50) we're on:
distractor_idx=$(( (IDX - 1) / TASKS_PER_DIST ))   # integer division → 0,1,2,3
inner_idx=$(( (IDX - 1) % TASKS_PER_DIST ))        # remainder → 0 … 89

# Now within that distractor block, split into 3 step‐sizes × 30 seeds:
run_idx=$(( (inner_idx % 30) + 1 ))   # 1 … 30
step_idx=$(( inner_idx / 30 ))        # integer division → 0,1,2

# List your four distractor values in matching order
DIST_VALUES=(0 10 20 50)
export NUM_DISTRACTOR=${DIST_VALUES[$distractor_idx]}

# List your three step‐sizes
STEPSIZES=(0.001 0.0001 0.00001)
export STEP_SIZE=${STEPSIZES[$step_idx]}

# Derive your seed however you like (here: mult of 1000)
export SEED=$(( run_idx * 1000 ))

export NAMETAG="distractors_${NUM_DISTRACTOR}_stepsize_${STEP_SIZE}"
echo ">>> TASK $IDX: run_idx=$run_idx, STEP_SIZE=$STEP_SIZE, NUM_DISTRACTOR=$NUM_DISTRACTOR, SEED=$SEED"

# ————— Run your training with those env vars —————
python -u main.py --config_path configs/config_a2c_train.py