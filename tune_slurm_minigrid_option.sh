#!/usr/bin/env bash
#SBATCH --job-name=Mask-both
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

mkdir -p logs

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
# Each distractor‐setting covers 30 seeds × 3 step‐sizes = 90 tasks
TASKS_PER_DIST=90

# Which distractor setting are we on? (0,1,2,3 → corresponds to 0,10,20,50)
distractor_idx=$(( (IDX - 1) / TASKS_PER_DIST ))    # integer division → 0,1,2,3
inner_idx=$(( (IDX - 1) % TASKS_PER_DIST ))         # remainder → 0 … 89

# Within each distractor block, split into 3 step‐sizes × 30 seeds:
step_idx=$(( inner_idx / 30 ))                      # integer division → 0,1,2
run_idx=$(( (inner_idx % 30) + 1 ))                  # 1 … 30

# List your four distractor values in matching order
DIST_VALUES=(0 10 20 50)
export NUM_DISTRACTOR=${DIST_VALUES[$distractor_idx]}

# List your three step‐sizes
STEPSIZES=(0.001 0.0001 0.00001)
export STEP_SIZE=${STEPSIZES[$step_idx]}

# Derive your seed (e.g., 1000, 2000, …, 30000)
export TMP_SEED=$(( run_idx * 1000 ))

# Other fixed options
export TMP_OPT="Mask"      # Mask, FineTune, DecWhole, Transfer
export MAX_NUM_OPTIONS=20      # 5, 10, 20
export MASK_TYPE="both"      # network, input, both

echo ">>> TASK $IDX: STEP_SIZE=$STEP_SIZE, SEED=$TMP_SEED, NUM_DISTRACTOR=$NUM_DISTRACTOR, OPT=$TMP_OPT"

# ————— Run your training with those env vars —————
python -u main.py --config_path configs/config_a2c_test_option.py