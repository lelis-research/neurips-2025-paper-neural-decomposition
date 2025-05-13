#!/usr/bin/env bash
set -euo pipefail

# choose how many you want to run at once:
JOB_LIMIT=32
seeds=( $(seq 60000 10000 300000) )
opts=(5 10)

# create a logs dir
mkdir -p logs

# for seed in "${seeds[@]}"; do
#   for max_opt in "${opts[@]}"; do
#     (
#       echo "→ [${seed}, ${max_opt}] starting"
#       # redirect stdout/stderr to per‐run log
#       TMP_SEED=$seed MAX_NUM_OPTIONS=$max_opt \
#         python main.py \
#         > "logs/seed_${seed}_opt${max_opt}_mask.out" \
#         2>&1
#       echo "→ [${seed}, ${max_opt}] done"
#     ) &
#     # throttle number of simultaneous jobs
#     if [[ $(jobs -r -p | wc -l) -ge $JOB_LIMIT ]]; then
#       wait -n
#     fi
#   done
# done

for seed in "${seeds[@]}"; do
  (
    export TMP_SEED=$seed
    echo "→ [${seed}] starting"
    python main.py \
      > "logs/seed_${seed}_transfer.out" 2>&1
    echo "→ [${seed}] done"
  ) &              # ← this ampersand backgrounds the whole subshell

  # throttle concurrency
  if [[ $(jobs -r -p | wc -l) -ge $JOB_LIMIT ]]; then
    wait -n      # wait for one job to finish before launching the next
  fi
done

# wait for the rest to finish
wait
echo "All experiments completed."