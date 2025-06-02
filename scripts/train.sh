#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-1:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --array=0-59  # 0-55

cd /home/rezaabdz/projects/def-lelis/rezaabdz/neurips-2025-paper-neural-decomposition


module load flexiblas
export FLEXIBLAS=blis2

source /home/rezaabdz/scratch/envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

wandb offline
OMP_NUM_THREADS=1 python -m pipelines.train_ppo --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID \
    --game_width=7 --total_timesteps=750000 --hidden_size=64 \
    --models_path_prefix="/home/rezaabdz/scratch/binary/models" \
    --learning_rate=0.0005 \
    --clip_coef=0.4 \
    --ent_coef=0.15 \
    # --learning_rate=0.001 \
    # --clip_coef=0.15 \
    # --ent_coef=0.05 \
    
    

# OMP_NUM_THREADS=1 python -m pipelines.train_ppo --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID \
#     --env_seeds=14
