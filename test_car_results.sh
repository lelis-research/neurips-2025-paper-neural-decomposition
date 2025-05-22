#!/bin/bash
#SBATCH --job-name=analyze-dqn
#SBATCH --output=logs/analyze-dqn-%A_%a.out
#SBATCH --error=logs/analyze-dqn-%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-lelis
#SBATCH --array=1-30

module load StdEnv/2020
module load gcc
module load flexiblas
module load python/3.10
module load mujoco/2.3.6

source /home/aghakasi/ENV/bin/activate

SEED=$(( SLURM_ARRAY_TASK_ID * 1000 ))

cd $HOME/scratch/neurips-2025-paper-neural-decomposition

# set your agent path prefix
export TEST_AGENT_PATH="car-train_${SEED}_4000000_all_actions"

# 1) car-train
export TEST_ENV_NAME="car-train"
OUT1=$(python main.py --config_path configs/config_car_ppo.py 2>&1)
S1=$(echo "$OUT1" \
     | grep -oP 'number of succesful park from 100:\s*\K\d+')

# 2) car-test
export TEST_ENV_NAME="car-test"
OUT2=$(python main.py --config_path configs/config_car_ppo.py 2>&1)
S2=$(echo "$OUT2" \
     | grep -oP 'number of succesful park from 100:\s*\K\d+')

# write “train, test” for this seed
mkdir -p Results_car_dqn_best_csv
echo "${S1},${S2}" > Results_car_nstepdqn_best_csv/results_${SEED}.csv