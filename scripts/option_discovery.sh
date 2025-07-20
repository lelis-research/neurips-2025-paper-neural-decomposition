#!/bin/bash
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:30:00
#SBATCH --output=selecting_options/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --array=0-69 #1080

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

#seeds=(10 20 30)
#reg_coefs=(0 0.01 0.05 0.1 0.25)

#num_seed=${#seeds[@]}
#num_reg_coefs=${#reg_coefs[@]}

#idx=$(( $SLURM_ARRAY_TASK_ID + 0 ))

#reg_index=$(( idx % num_reg_coefs ))
#idx=$(( idx / num_reg_coefs ))

#sd_index=$(( idx % num_seed ))

#SD="${seeds[${sd_index}]}"
#REG="${reg_coefs[${reg_index}]}"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
    --seed $SLURM_ARRAY_TASK_ID\
    --game_width 9\
    --cpus $SLURM_CPUS_PER_TASK\
    --option_mode "didec"\
    --reg_coef 0.01\
    --mask_type "both"
