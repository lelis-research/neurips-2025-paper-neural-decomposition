#!/usr/bin/env bash
#SBATCH --job-name=param_sweep
#SBATCH --time=7:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5              # each job is single-threaded
#SBATCH --account=aip-lelis
#SBATCH --array=0-624        
#SBATCH --output=logs/param_sweep_%A_%a.out
#SBATCH --error=logs/param_sweep_%A_%a.err

set -euo pipefail

# ——————— Load Compute Canada modules ———————
module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

# ————— Activate your Python environment —————
source /home/aghakasi/ENV/bin/activate

# make sure the logs/ folder exists
mkdir -p logs

# ————— Generate 5 grid points for each parameter —————
read -r -a step_size_list <<< "$(
  python - << 'EOF'
import numpy as np
print(" ".join(map(str, np.logspace(np.log10(3e-5), np.log10(3e-3), 5))))
EOF
)"

read -r -a num_minibatches_list <<< "$(
  python - << 'EOF'
import numpy as np
print(" ".join(map(str, np.linspace(16, 128, 5, dtype=int))))
EOF
)"

read -r -a rollout_steps_list <<< "$(
  python - << 'EOF'
import numpy as np
print(" ".join(map(str, np.linspace(500, 5000, 5, dtype=int))))
EOF
)"

read -r -a entropy_coef_list <<< "$(
  python - << 'EOF'
import numpy as np
print(" ".join(map(str, np.linspace(0, 0.2, 5))))
EOF
)"

# ————— Compute indices for this array task —————
TOTAL_NUM_MINI=${#num_minibatches_list[@]}
TOTAL_ROLL=${#rollout_steps_list[@]}
TOTAL_ENT=${#entropy_coef_list[@]}
TOTAL_COMB=$(( TOTAL_NUM_MINI * TOTAL_ROLL * TOTAL_ENT ))

IDX=$SLURM_ARRAY_TASK_ID

ss_idx=$(( IDX / TOTAL_COMB ))
rem0=$(( IDX % TOTAL_COMB ))
nm_idx=$(( rem0 / (TOTAL_ROLL * TOTAL_ENT) ))
rem1=$(( rem0 % (TOTAL_ROLL * TOTAL_ENT) ))
rs_idx=$(( rem1 / TOTAL_ENT ))
ec_idx=$(( rem1 % TOTAL_ENT ))

# ————— Export for config_sweep.py to pick up —————
export STEP_SIZE="${step_size_list[$ss_idx]}"
export NUM_MINIBATCHES="${num_minibatches_list[$nm_idx]}"
export ROLLOUT_STEPS="${rollout_steps_list[$rs_idx]}"
export ENTROPY_COEF="${entropy_coef_list[$ec_idx]}"
export NAMETAG="ss_${step_size_list[$ss_idx]}_m_${num_minibatches_list[$nm_idx]}_r_${rollout_steps_list[$rs_idx]}_e_${entropy_coef_list[$ec_idx]}"

echo "Running combination $IDX: \
step_size=$STEP_SIZE, \
num_minibatches=$NUM_MINIBATCHES, \
rollout_steps=$ROLLOUT_STEPS, \
entropy_coef=$ENTROPY_COEF"

# ————— Run your training/tuning script —————
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_sweep.py