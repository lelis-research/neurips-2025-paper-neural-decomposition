#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-1:30
#SBATCH --output=%j-%N.out
#SBATCH --account=rrg-lelis
#SBATCH --array=0-149  # 2 lr × 5 clip × 5 ent × 3 seeds = 150 jobs

cd /home/rezaabdz/projects/def-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

module load flexiblas
export FLEXIBLAS=blis2

source /home/rezaabdz/scratch/envs/venv/bin/activate

# Final parameter ranges
learning_rates=(0.0005 0.001)              # 2
clip_coefs=(0.1 0.15 0.2 0.3 0.4)          # 5
ent_coefs=(0.01 0.05 0.1 0.15 0.2)         # 5

num_seeds=3
total_experiments=$((2 * 5 * 5))  # 50

global_idx=$SLURM_ARRAY_TASK_ID
exp_idx=$((global_idx / num_seeds))  # 0–49
seed=$((global_idx % num_seeds))     # 0–2

# Indexing
lr_idx=$((exp_idx / 25))             # 0–1
cc_idx=$(((exp_idx % 25) / 5))       # 0–4
ec_idx=$((exp_idx % 5))              # 0–4

lr=${learning_rates[$lr_idx]}
clip=${clip_coefs[$cc_idx]}
ent=${ent_coefs[$ec_idx]}

echo "Running job $global_idx (exp=$exp_idx, seed=$seed) with:"
echo "  Learning rate: $lr"
echo "  Clip coef: $clip"
echo "  Entropy coef: $ent"
echo "  Seed: $seed"

OMP_NUM_THREADS=1 python -m pipelines.train_ppo \
  --seed=$seed \
  --env_seeds=0 \
  --game_width=7 --total_timesteps=1250000 --hidden_size=6 \
  --learning_rate=$lr \
  --clip_coef=$clip \
  --ent_coef=$ent \
  --models_path_prefix="binary/models/parameter_sweep" \
  --param_sweep