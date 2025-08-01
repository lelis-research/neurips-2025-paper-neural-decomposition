#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:40:00
#SBATCH --output=fourrooms-sweep-2/%A-%a.out
#SBATCH --account=rrg-lelis
#SBATCH --mail-type=ALL
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --array=0-5399 #1080

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"


# seeds=(6 7 8 12 13 14 15 16 17) #9
seeds=(0 1 2 3 4 5) #6
learning_rates=(0.01 0.005 0.001 0.0005 0.00005) #5
clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3) #6
ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2) #6
num_steps=(0.0 0.01 0.05 0.1 0.25)


num_seed=${#seeds[@]}
num_lr=${#learning_rates[@]}
num_ent_coef=${#ent_coefs[@]}
num_clip_coef=${#clip_coef[@]}
num_s=${#num_steps[@]}

#idx=$SLURM_ARRAY_TASK_ID
idx=$(( $SLURM_ARRAY_TASK_ID + 0 ))


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
    --num_steps 2000\
    --clip_coef "${CLIP}"\
    --env_id "MiniGrid-FourRooms-v0"\
    --game_width 9\
    --total_timesteps 1000000\
    --save_run_info 1\
    --method "options"\
    --option_mode "didec"\
    --reg_coef "${NUM}"\
    --mask_type "internal"\
    --sweep_run 0
