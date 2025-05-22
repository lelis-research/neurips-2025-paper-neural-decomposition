#!/usr/bin/env bash
#SBATCH --job-name=param_sweep_ddpg
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10
#SBATCH --account=aip-lelis
#SBATCH --array=0-323                   # 3×3×3×2×3×2 = 324 combos
#SBATCH --output=logs/param_sweep_ddpg_%A_%a.out
#SBATCH --error=logs/param_sweep_ddpg_%A_%a.err

set -euo pipefail

# ————— Load modules —————
module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

# ————— Activate environment —————
source /home/aghakasi/ENV/bin/activate

mkdir -p logs

# ————— Static grid definitions —————
actor_lr_list=(0.0001 0.001 0.01)
critic_lr_list=(0.0001 0.001 0.01)
buf_size_list=(100000 500000 1000000)
batch_size_list=(64 128)
ou_theta_list=(0.15 0.2 0.25)
ou_sigma_list=(0.1 0.2)

# ————— Compute sizes —————
TOTAL_AL=${#actor_lr_list[@]}   # 3
TOTAL_CL=${#critic_lr_list[@]}  # 3
TOTAL_BUF=${#buf_size_list[@]}  # 3
TOTAL_BS=${#batch_size_list[@]} # 2
TOTAL_TH=${#ou_theta_list[@]}   # 3
TOTAL_SIG=${#ou_sigma_list[@]}  # 2

# — combinations per actor‐lr slice —————
PER_AL=$(( TOTAL_CL * TOTAL_BUF * TOTAL_BS * TOTAL_TH * TOTAL_SIG ))  # 3×3×2×3×2 = 324/3 = 108

IDX=$SLURM_ARRAY_TASK_ID

# — decode indices —————
al_idx=$(( IDX / PER_AL ))
rem0=$(( IDX % PER_AL ))

cl_idx=$(( rem0 / (TOTAL_BUF * TOTAL_BS * TOTAL_TH * TOTAL_SIG) ))
rem1=$(( rem0 % (TOTAL_BUF * TOTAL_BS * TOTAL_TH * TOTAL_SIG) ))

buf_idx=$(( rem1 / (TOTAL_BS * TOTAL_TH * TOTAL_SIG) ))
rem2=$(( rem1 % (TOTAL_BS * TOTAL_TH * TOTAL_SIG) ))

bs_idx=$(( rem2 / (TOTAL_TH * TOTAL_SIG) ))
rem3=$(( rem2 % (TOTAL_TH * TOTAL_SIG) ))

th_idx=$(( rem3 / TOTAL_SIG ))
sig_idx=$(( rem3 % TOTAL_SIG ))

# — export for config to pick up —————
export ACTOR_LR="${actor_lr_list[$al_idx]}"
export CRITIC_LR="${critic_lr_list[$cl_idx]}"
export BUF_SIZE="${buf_size_list[$buf_idx]}"
export BATCH_SIZE="${batch_size_list[$bs_idx]}"
export OU_THETA="${ou_theta_list[$th_idx]}"
export OU_SIGMA="${ou_sigma_list[$sig_idx]}"

export NAMETAG="\
al_${ACTOR_LR}_cl_${CRITIC_LR}_buf_${BUF_SIZE}_bs_${BATCH_SIZE}\
_th_${OU_THETA}_sig_${OU_SIGMA}"

echo "Combo $IDX → \
actor_lr=$ACTOR_LR, critic_lr=$CRITIC_LR, buf_size=$BUF_SIZE, \
batch_size=$BATCH_SIZE, ou_theta=$OU_THETA, ou_sigma=$OU_SIGMA"

# — run DDPG training —————
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_car_ddpg.py