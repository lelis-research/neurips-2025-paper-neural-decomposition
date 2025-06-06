#!/usr/bin/env bash
#SBATCH --time=0-00:20:00  #  0-00:10:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=32
#SBATCH --account=rrg-lelis
#SBATCH --array=0-14
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# Move into repo
cd /home/rezaabdz/projects/def-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

# Load modules & env
module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
source /home/rezaabdz/scratch/envs/venv2/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…300

# Map to agent‐seed (0…14) and env‐seed (0…3)
# AGENT_IDX=$(( (IDX) / 4 ))
export SEED=$((IDX))
export ENV_SEED=12
export NAMETAG="env_${ENV_SEED}"
# export TMP_SEED=$(( AGENT_IDX * 1000 )) 

export GAME_WIDTH=5
export HIDDEN_SIZE=6
export MASK_TYPE="network"
export TMP_OPT="DecOption" # Mask, FineTune, DecWhole, Transfer, DecOption
export TOTAL_STEPS=200000
export STEP_SIZE=0.0003

export MODE="train_option" # train, test, plot, tune, train_option, test_option, search_option

# Run your script (it should read both $SEED and $ENV_SEED from os.environ)
python -u main.py --config_path configs/combogrid/config_a2c_option.py