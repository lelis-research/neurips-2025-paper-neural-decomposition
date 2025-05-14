#!/bin/bash
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=5G       # memory per node
#SBATCH --time=1:30:00      # time (DD-HH:MM)
#SBATCH --output=outputsSweepcomboprimAsync4x4/%A-%a.out  # %N for node name, %j for jobID
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=aip-lelis
#SBATCH --array=0-1000


source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


seeds=(1 2 3)
learning_rates=(0.01 0.005 0.001 0.0005)
clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)
ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)
max_length=(30)
visit_bonus=(1)
use_option=(0)
env_id=(0 1 2 3)


num_seed=${#seeds[@]}
num_lr=${#learning_rates[@]}
num_ent_coef=${#ent_coefs[@]}
num_clip_coef=${#clip_coef[@]}
num_max_len=${#max_length[@]}
num_visit=${#visit_bonus[@]}
num_option=${#use_option[@]}
num_env_id=${#env_id[@]}

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

# Get index for max len
max_len_index=$(( idx % num_max_len ))
idx=$(( idx / num_max_len ))

# Get index for visit bonus
visit_index=$(( idx % num_visit ))
idx=$(( idx / num_visit ))

# Get index for use option
option_index=$(( idx % num_option ))
idx=$(( idx / num_option ))

# Get index for env id
env_id_index=$(( idx % num_env_id ))
idx=$(( idx / num_env_id ))

# Get index for seed
sd_index=$(( idx % num_seed ))

SD="${seeds[${sd_index}]}"
LR="${learning_rates[${lr_index}]}"
ENT="${ent_coefs[${ent_index}]}"
CLIP="${clip_coef[${clip_index}]}"
MAXLEN="${max_length[${max_len_index}]}"
VISIT="${visit_bonus[${visit_index}]}"
OPTION="${use_option[${option_index}]}"
ENVID="${env_id[${env_id_index}]}"

OMP_NUM_THREADS=1 python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed "${SD}" \
    --actor_lr "${LR}"\
    --critic_lr "${LR}"\
    --max_episode_length "${MAXLEN}"\
    --ent_coef "${ENT}"\
    --clip_coef "${CLIP}"\
    --visitation_bonus "${VISIT}"\
    --use_options "${OPTION}"\
    --env_seed "${ENVID}"\
    --env_id "ComboGrid" \
    --game_width 4\
    --total_timesteps 1000000\
    --processed_options 0\
    --save_run_info 1
