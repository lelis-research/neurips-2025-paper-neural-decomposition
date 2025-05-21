#!/usr/bin/env bash
#SBATCH --job-name=param_sweep_dqn
#SBATCH --time=7:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10              # each job is single-threaded
#SBATCH --account=aip-lelis
#SBATCH --array=0-323                   # 3×3×3×2×2×3 = 324 combos
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

# ————— Static grid definitions —————
step_size_list=(0.01 0.001 0.0001)
batch_size_list=(64 128 256)
target_update_freq_list=(100 500 1000)
epsilon_list=(0.01 0.1)
replay_buffer_cap_list=(1000000 2000000)
action_res_list=(3 5 7)

# ————— Compute lengths for indexing —————
TOTAL_SS=${#step_size_list[@]}           # 3
TOTAL_BS=${#batch_size_list[@]}          # 3
TOTAL_TU=${#target_update_freq_list[@]}  # 3
TOTAL_EPS=${#epsilon_list[@]}            # 2
TOTAL_RBC=${#replay_buffer_cap_list[@]}  # 2
TOTAL_AR=${#action_res_list[@]}          # 3

# ————— Combinations per step_size slice —————
PER_SS=$(( TOTAL_BS * TOTAL_TU * TOTAL_EPS * TOTAL_RBC * TOTAL_AR ))  # 3×3×2×2×3 = 108

IDX=$SLURM_ARRAY_TASK_ID

# ————— Decode each index —————
ss_idx=$(( IDX / PER_SS ))
rem0=$(( IDX % PER_SS ))

bs_idx=$(( rem0 / (TOTAL_TU * TOTAL_EPS * TOTAL_RBC * TOTAL_AR) ))
rem1=$(( rem0 % (TOTAL_TU * TOTAL_EPS * TOTAL_RBC * TOTAL_AR) ))

tu_idx=$(( rem1 / (TOTAL_EPS * TOTAL_RBC * TOTAL_AR) ))
rem2=$(( rem1 % (TOTAL_EPS * TOTAL_RBC * TOTAL_AR) ))

eps_idx=$(( rem2 / (TOTAL_RBC * TOTAL_AR) ))
rem3=$(( rem2 % (TOTAL_RBC * TOTAL_AR) ))

rbc_idx=$(( rem3 / TOTAL_AR ))
ar_idx=$(( rem3 % TOTAL_AR ))

# ————— Export for your config to pick up —————
export STEP_SIZE="${step_size_list[$ss_idx]}"
export BATCH_SIZE="${batch_size_list[$bs_idx]}"
export TARGET_UPDATE_FREQ="${target_update_freq_list[$tu_idx]}"
export EPSILON="${epsilon_list[$eps_idx]}"
export REPLAY_BUFFER_CAP="${replay_buffer_cap_list[$rbc_idx]}"
export ACTION_RES="${action_res_list[$ar_idx]}"

export NAMETAG="\
ss_${STEP_SIZE}_bs_${BATCH_SIZE}_tu_${TARGET_UPDATE_FREQ}\
_e_${EPSILON}_rb_${REPLAY_BUFFER_CAP}_ar_${ACTION_RES}"

echo "Running combo $IDX → \
step_size=$STEP_SIZE, batch_size=$BATCH_SIZE, \
target_update_freq=$TARGET_UPDATE_FREQ, epsilon=$EPSILON, \
replay_buffer_cap=$REPLAY_BUFFER_CAP, action_res=$ACTION_RES"

# ————— Run your training/tuning script —————
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_car_dqn.py