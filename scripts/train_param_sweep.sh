#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:40:00
#SBATCH --output=unlock-sweep/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=1-1000 #1080

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

seeds=(0 1 3 4 6 7) #6
learning_rates=(0.005 0.001 0.0001 0.0005 0.00005) #5
clip_coef=(0.1 0.15 0.2 0.3 0.5) #5
ent_coefs=(0.01 0.02 0.03 0.05 0.1) #5
num_steps=(128 500 1000 2000) #4


num_seed=${#seeds[@]}
num_lr=${#learning_rates[@]}
num_ent_coef=${#ent_coefs[@]}
num_clip_coef=${#clip_coef[@]}
num_s=${#num_steps[@]}

#idx=$SLURM_ARRAY_TASK_ID
idx=$(( $SLURM_ARRAY_TASK_ID + 2000 ))


# Get index for learning rate
lr_index=$(( idx % num_lr ))
idx=$(( idx / num_lr ))

# Get index for entropy coef
ent_index=$(( idx % num_ent_coef ))
idx=$(( idx / num_ent_coef ))

# Get index for clip coef
clip_index=$(( idx % num_clip_coef ))
idx=$(( idx / num_clip_coef ))

num_index=$(( idx % num_s ))
idx=$(( idx / num_s ))


# Get index for seed
sd_index=$(( idx % num_seed ))

SD="${seeds[${sd_index}]}"
LR="${learning_rates[${lr_index}]}"
ENT="${ent_coefs[${ent_index}]}"
CLIP="${clip_coef[${clip_index}]}"
NUM="${num_steps[${num_index}]}"

OMP_NUM_THREADS=1 python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed "${SD}" \
    --learning_rate "${LR}"\
    --ent_coef "${ENT}"\
    --num_steps "${NUM}"\
    --clip_coef "${CLIP}"\
    --env_id "MiniGrid-Unlock-v0"\
    --game_width 9\
    --total_timesteps 1000000\
    --save_run_info 1\
    --method "no_options"\
    --option_mode "vanilla"