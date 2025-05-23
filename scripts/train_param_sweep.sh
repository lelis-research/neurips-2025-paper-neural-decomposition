#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-1:00
#SBATCH --output=%j-%N.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-239  #0-719  # 240 experiments × 3 seeds = 720 jobs

cd /home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition


module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate

# Parameter ranges
learning_rates=(0.0005 0.001 0.005 0.01 0.05)      # 5
clip_coefs=(0.1 0.15 0.2 0.3 0.4 0.5)              # 6
ent_coefs=(0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35)   # 8

num_seeds=3
total_experiments=$((5 * 6 * 8))  # 240

global_idx=$SLURM_ARRAY_TASK_ID
exp_idx=$((global_idx / num_seeds))  # 0–239
seed=$((global_idx % num_seeds))     # 0–2

lr_idx=$((exp_idx / 48))             # 0–4
cc_idx=$(((exp_idx % 48) / 8))       # 0–5
ec_idx=$((exp_idx % 8))              # 0–7

lr=${learning_rates[$lr_idx]}
clip=${clip_coefs[$cc_idx]}
ent=${ent_coefs[$ec_idx]}

echo "Running job $global_idx (exp=$exp_idx, seed=$seed) with:"
echo "  Learning rate: $lr"
echo "  Clip coef: $clip"
echo "  Entropy coef: $ent"
echo "  Seed: $seed"

OMP_NUM_THREADS=1 python -m pipelines.train_ppo \
  --seed=2 \
  --learning_rate=$lr \
  --clip_coef=$clip \
  --ent_coef=$ent \
  --game_width=8 \
  --hidden_size=6 \
  --models_path_prefix="binary/models/parameter_sweep"