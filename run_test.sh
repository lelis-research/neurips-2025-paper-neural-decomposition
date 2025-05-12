#!/usr/bin/env bash
set -euo pipefail

# max concurrent jobs
JOB_LIMIT=32

# parameter grids
seeds=(10000 20000 30000 40000 50000)
opts=(5 10)
envs=("Medium_Maze" "Large_Maze" "Hard_Maze")

# make sure your config.py reads these three from env:
#   TMP_SEED, MAX_NUM_OPTIONS, TEST_OPTION_ENV_NAME
# e.g. in config.py:
#   tmp_seed        = int(os.environ.get("TMP_SEED", 10000))
#   max_num_options = int(os.environ.get("MAX_NUM_OPTIONS", 5))
#   test_option_env_name = os.environ.get("TEST_OPTION_ENV_NAME", "Hard_Maze")

mkdir -p logs

for seed in "${seeds[@]}"; do
  for opt in "${opts[@]}"; do
    for env in "${envs[@]}"; do
      (
        export TMP_SEED=$seed
        export MAX_NUM_OPTIONS=$opt
        export TEST_OPTION_ENV_NAME=$env

        echo "→ [seed=${seed}, opt=${opt}, env=${env}] starting"
        python main.py \
          > "logs/test_options_seed_${seed}_opt${opt}_env${env}.log" 2>&1
        echo "→ [seed=${seed}, opt=${opt}, env=${env}] done"
      ) &

      # throttle to $JOB_LIMIT concurrent jobs
      if [[ $(jobs -r -p | wc -l) -ge $JOB_LIMIT ]]; then
        wait -n
      fi
    done
  done
done

# wait for any remaining jobs
wait
echo "All experiments completed."