#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=6G
#SBATCH --time=0-8:00
#SBATCH --output=%j-%N.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-719

module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate

# Parameter ranges
learning_rates=(0.0005 0.001 0.005 0.01 0.05)      # 5
clip_coefs=(0.1 0.15 0.2 0.3 0.4 0.5)              # 6
ent_coefs=(0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35)   # 8

total_experiments=$((5 * 6 * 8))  # 240

idx=$SLURM_ARRAY_TASK_ID
seed=$((idx % 3))
exp_idx=$((idx / 3))  # 0 to 239

lr_idx=$((exp_idx / 48))               # 0–4
cc_idx=$(((exp_idx % 48) / 8))         # 0–5
ec_idx=$((exp_idx % 8))                # 0–7

lr=${learning_rates[$lr_idx]}
clip=${clip_coefs[$cc_idx]}
ent=${ent_coefs[$ec_idx]}

echo "Running job $idx with:"
echo "  Seed: $seed"
echo "  Learning rate: $lr"
echo "  Clip coef: $clip"
echo "  Entropy coef: $ent"

OMP_NUM_THREADS=1 python -m pipelines.parameter_sweep_testing \
  --seed=$seed \
  --learning_rate=$lr \
  --clip_coef=$clip \
  --ent_coef=$ent