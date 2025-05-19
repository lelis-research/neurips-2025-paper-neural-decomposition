#!/usr/bin/env bash
#SBATCH --job-name=no_option
#SBATCH --time=0-03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10
#SBATCH --account=aip-lelis
#SBATCH --output=logs/no_option_%j.out    # %j → SLURM job ID
#SBATCH --error=logs/no_option_%j.err

# make sure the logs/ folder exists
mkdir -p logs

set -euo pipefail

# ——————— Load Compute Canada modules ———————
module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

# ————— Activate your Python environment —————
source /home/aghakasi/ENV/bin/activate

# unbuffer Python so print() shows up immediately
export PYTHONUNBUFFERED=1  
export FLEXIBLAS=imkl

# ————— Run your script —————
export #TMP_SEED
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_car.py