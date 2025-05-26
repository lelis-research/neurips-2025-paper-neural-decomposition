#!/usr/bin/env bash
#SBATCH --job-name=param_sweep_l1_all
#SBATCH --time=7:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=15              # each job is single-threaded
#SBATCH --account=aip-lelis
#SBATCH --array=0-242        
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

# ————— Static grid definitions (3 values each) —————
step_size_list=(3e-5   1e-4    3e-4)
num_minibatches_list=(32 64 128)
rollout_steps_list=(2000 4000 6000)
entropy_coef_list=(0.0 0.1 0.2)
l1_lambda_list=(1e-5 1e-4 1e-3)

# ————— Compute lengths for indexing —————
TOTAL_SS=${#step_size_list[@]}
TOTAL_NM=${#num_minibatches_list[@]}
TOTAL_RS=${#rollout_steps_list[@]}
TOTAL_ENT=${#entropy_coef_list[@]}
TOTAL_L1=${#l1_lambda_list[@]}

# ————— Total combos *per* step_size slice —————
PER_SS=$(( TOTAL_NM * TOTAL_RS * TOTAL_ENT * TOTAL_L1 ))  # 3⁴ = 81

IDX=$SLURM_ARRAY_TASK_ID

# ————— Decode each index —————
ss_idx=$(( IDX / PER_SS ))
rem0=$(( IDX % PER_SS ))

nm_idx=$(( rem0 / (TOTAL_RS * TOTAL_ENT * TOTAL_L1) ))
rem1=$(( rem0 % (TOTAL_RS * TOTAL_ENT * TOTAL_L1) ))

rs_idx=$(( rem1 / (TOTAL_ENT * TOTAL_L1) ))
rem2=$(( rem1 % (TOTAL_ENT * TOTAL_L1) ))

ent_idx=$(( rem2 / TOTAL_L1 ))
l1_idx=$(( rem2 % TOTAL_L1 ))

# ————— Export for your config_sweep.py to pick up —————
export STEP_SIZE="${step_size_list[$ss_idx]}"
export NUM_MINIBATCHES="${num_minibatches_list[$nm_idx]}"
export ROLLOUT_STEPS="${rollout_steps_list[$rs_idx]}"
export ENTROPY_COEF="${entropy_coef_list[$ent_idx]}"
export L1_LAMBDA="${l1_lambda_list[$l1_idx]}"

export NAMETAG="\
ss_${STEP_SIZE}_m_${NUM_MINIBATCHES}_r_${ROLLOUT_STEPS}\
_e_${ENTROPY_COEF}_l1_${L1_LAMBDA}"

echo "Running combo $IDX → \
step_size=$STEP_SIZE, num_minibatches=$NUM_MINIBATCHES, \
rollout_steps=$ROLLOUT_STEPS, entropy_coef=$ENTROPY_COEF, \
l1_lambda=$L1_LAMBDA"

# ————— Run your training/tuning script —————
python main.py --config_path ~/scratch/neurips-2025-paper-neural-decomposition/configs/config_sweep.py