#!/usr/bin/env bash
#SBATCH --job-name=mask-exp
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --time=0-20:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-49
#SBATCH --account=aip-lelis


set -euo pipefail

# Load modules
module load mujoco opencv python  # adjust as needed

cd /home/aghakasi/scratch/neurips-2025-paper-neural-decomposition
source myenv/bin/activate

# Generate seeds and opts
seeds=( $(seq 60000 10000 300000) )
# seeds=(140000 180000 210000 260000)
opts=(5 10)

# Create all combinations
combo_index=$SLURM_ARRAY_TASK_ID
seed_index=$((combo_index / ${#opts[@]}))
opt_index=$((combo_index % ${#opts[@]}))

seed=${seeds[$seed_index]}
max_opt=${opts[$opt_index]}

echo "→ [${seed}, ${max_opt}] starting"

TMP_SEED=$seed MAX_NUM_OPTIONS=$max_opt \
  python main.py > logs/seed_${seed}_opt${max_opt}_mask.out 2>&1

echo "→ [${seed}, ${max_opt}] done"