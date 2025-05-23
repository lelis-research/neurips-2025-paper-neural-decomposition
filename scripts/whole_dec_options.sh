#!/bin/bash
#SBATCH --cpus-per-task=64   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=1G        # memory per node
#SBATCH --time=0-1:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID
#SBATCH --account=aip-lelis
#SBATCH --array=0-14

cd /home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

OMP_NUM_THREADS=1 python -m pipelines.dec_option --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID
