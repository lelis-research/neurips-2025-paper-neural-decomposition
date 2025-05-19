#!/usr/bin/env bash
#SBATCH --job-name=finetune
#SBATCH --time=0-03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --account=aip-lelis

# Run seeds 1 through 10 → we'll multiply by 1000 below
#SBATCH --array=1-7,9-10

#SBATCH --output=logs/finetune_%A_%a.out    # %A = job ID, %a = array index
#SBATCH --error=logs/finetune_%A_%a.err

# ensure log directory exists
mkdir -p logs

set -euo pipefail

# ——————— Load Compute Canada modules ———————
module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

# ————— Activate your Python environment ——————
source /home/aghakasi/ENV/bin/activate

# unbuffer Python so your print()s show up immediately
export PYTHONUNBUFFERED=1  
export FLEXIBLAS=imkl

# ————— Set the seed for this array task ——————
# this will give TMP_SEED=1000,2000,…,10000
export TMP_SEED=$(( SLURM_ARRAY_TASK_ID * 1000 ))
export MAX_NUM_OPTIONS=10
echo "Running with TMP_SEED=$TMP_SEED MAX_NUM_OPTIONS=$MAX_NUM_OPTIONS"

# ————— Run your script ——————
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/config_finetune.py