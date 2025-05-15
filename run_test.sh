#!/opt/homebrew/bin/bash
set -euo pipefail

# max concurrent jobs
JOB_LIMIT=16

# parameter grids
seeds=( $(seq 10000 10000 300000) )
opts=(5)
envs=("Large_Maze")
step_sizes=(0.0003)
ent_coeffs=(0.0)
tmp_opt="Mask"
# make sure your config.py reads these three from env:
#   TMP_SEED, MAX_NUM_OPTIONS, TEST_OPTION_ENV_NAME
# e.g. in config.py:
#   tmp_seed        = int(os.environ.get("TMP_SEED", 10000))
#   max_num_options = int(os.environ.get("MAX_NUM_OPTIONS", 5))
#   test_option_env_name = os.environ.get("TEST_OPTION_ENV_NAME", "Hard_Maze")

mkdir -p logs

# *************************** For Mask, DecWhole, FineTuning Baseline
for seed in "${seeds[@]}"; do
  for opt in "${opts[@]}"; do
    for i in "${!envs[@]}"; do
      (
        env="${envs[$i]}"
        stepsize="${step_sizes[$i]}"
        entcoeff="${ent_coeffs[$i]}"

        export TMP_SEED=$seed
        export TEST_OPTION_ENV_NAME=$env
        export STEP_SIZE=$stepsize
        export ENTROPY_COEF=$entcoeff
        export TMP_OPT=$tmp_opt
        export MAX_NUM_OPTIONS=$opt

        echo "→ [seed=${seed}, env=${env}, stepsize=${stepsize}, entcoeff=${entcoeff}] starting"
        python main.py \
        > "logs/test_options_seed_${tmp_opt}_${opt}_${seed}_env${env}.log" 2>&1
      echo "→ [seed=${seed}, env=${env}] done"
        echo "→ [seed=${seed}, opt=${opt}, env=${env}] done"
      ) &

      # throttle to $JOB_LIMIT concurrent jobs
      if [[ $(jobs -r -p | wc -l) -ge $JOB_LIMIT ]]; then
        wait -n
      fi
    done
  done
done

# *************************** For Transfer Baseline
# for seed in "${seeds[@]}"; do
#   for i in "${!envs[@]}"; do
#     (
#       env="${envs[$i]}"
#       stepsize="${step_sizes[$i]}"
#       entcoeff="${ent_coeffs[$i]}"

#       export TMP_SEED=$seed
#       export TEST_OPTION_ENV_NAME=$env
#       export STEP_SIZE=$stepsize
#       export ENTROPY_COEF=$entcoeff
#       export TMP_OPT=$tmp_opt

#       echo "→ [seed=${seed}, env=${env}, stepsize=${stepsize}, entcoeff=${entcoeff}] starting"
#       python main.py \
#         > "logs/test_options_seed_${seed}_env${env}.log" 2>&1
#       echo "→ [seed=${seed}, env=${env}] done"
#     ) &

#     if [[ $(jobs -r -p | wc -l) -ge $JOB_LIMIT ]]; then
#       wait -n
#     fi
#   done
# done


# # wait for any remaining jobs
# wait
# echo "All experiments completed."