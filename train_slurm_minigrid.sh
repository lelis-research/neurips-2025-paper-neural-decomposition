#!/usr/bin/env bash
#SBATCH --job-name=minigrid
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-lelis
#SBATCH --array=1-300      
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

set -euo pipefail

# Move into repo
cd ~/scratch/neurips-2025-paper-neural-decomposition

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
IDX=$SLURM_ARRAY_TASK_ID   # 1…300

# Map to agent‐seed (1…30) and env‐seed (1…10)
AGENT_IDX=$(( (IDX - 1) / 10 + 1 ))
export SEED=$(( AGENT_IDX * 1000 ))
export ENV_SEED=$(( (IDX - 1) % 10 + 1 ))
export NAMETAG="env_${ENV_SEED}"
# export TMP_SEED=$(( AGENT_IDX * 1000 )) 


# Run your script (it should read both $SEED and $ENV_SEED from os.environ)
python -u main.py --config_path configs/config_a2c.py