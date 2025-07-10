#!/bin/bash
#SBATCH --cpus-per-task=48   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-4:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --array=0

cd /home/rezaabdz/projects/def-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

module load flexiblas
export FLEXIBLAS=blis2

source /home/rezaabdz/scratch/envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

OMP_NUM_THREADS=1 python -m pipelines.option_discovery --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID \
    --mask_type="input" \
    --cache_path="/home/rezaabdz/scratch/binary/cache/" \
    --game_width=6 --hidden_size=64
