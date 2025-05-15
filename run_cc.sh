#!/usr/bin/env bash
#SBATCH --job-name=tune-exp
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=0-20:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-5
#SBATCH --account=aip-lelis

set -euo pipefail

# Load necessary modules
module load mujoco opencv python

cd /home/aghakasi/scratch/neurips-2025-paper-neural-decomposition
source myenv/bin/activate

# Define grid
# DecWhole 140000 180000 210000 260000 280000
# Mask 180000 210000
seeds=(180000 210000 260000)
opts=(5 10)
baseline="tune"

# Get indices
combo_index=$SLURM_ARRAY_TASK_ID
seed_index=$((combo_index / ${#opts[@]}))
opt_index=$((combo_index % ${#opts[@]}))

seed=${seeds[$seed_index]}
max_opt=${opts[$opt_index]}
option_exp_name="Options_FineTune_Maze_m_Seed_${seed}"  

echo "→ [${seed}, ${max_opt}] starting"

export TMP_SEED=$seed
export MAX_NUM_OPTIONS=$max_opt
export BASELINE=$baseline
export OPTION_EXP_NAME=$option_exp_name  

python main.py > logs/seed_${seed}_opt${max_opt}_${baseline}.out 2>&1

echo "→ [${seed}, ${max_opt}] done"