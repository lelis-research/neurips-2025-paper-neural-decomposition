#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --output=dec_whole/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=47,49,50,51,52,53,54,57,58,61,64,67,70,72,74,75,76,77,80,81,82,83,84,86,88,89,90,91,94,95

#0,2,3,4,5,6,13,14,16,17,18,19,21,22,23,24,26,28,30,31,32,33,35,36,37,39,41,42,43,44
#47,49,50,51,52,53,54,57,58,61,64,67,70,72,74,75,76,77,80,81,82,83,84,86,88,89,90,91,94,95

#0,2,3,4,6,7,8,11,14,16,19,21,25,27,28,29,30,33,34,35,36,40,42,43,44,45,49,51,52,53

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/algorithms/fine_tuning.py --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID

#Used foe Dec-Whole
# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID --option_mode="dec-whole"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/base_policy_transferred.py --seed=$SLURM_ARRAY_TASK_ID


