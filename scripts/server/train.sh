#!/bin/bash

CUR_PATH=$(pwd)

# Activate your environment
cd /home/abreza/neurips-2025-paper-neural-decomposition
# module load flexiblas
# export FLEXIBLAS=blis2

source envs/venv/bin/activate

wandb offline

# Define the command template
run_job() {
  SEED=$1
  OMP_NUM_THREADS=1 python -m pipelines.train_ppo \
    --cpus=1 \
    --seed=$SEED \
    --game_width=6 \
    --total_timesteps=1500000 \
    --hidden_size=6 \
    > $CUR_PATH/logs/job_$SEED.out 2>&1
}

# Export function for parallel
export -f run_job

# Create logs directory
mkdir -p $CUR_PATH/logs

# Run jobs 0 through 59 with max 32 in parallel
parallel -j 64 run_job ::: $(seq 0 59)