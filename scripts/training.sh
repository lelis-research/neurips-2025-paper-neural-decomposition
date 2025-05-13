#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=1G       # memory per node
#SBATCH --time=7:00:00      # time (DD-HH:MM)
#SBATCH --output=outputsCombo0Prim/%A-%a.out  # %N for node name, %j for jobID
#SBATCH --account=aip-lelis
#SBATCH --no-requeue
#SBATCH --array=21,23,24,25,28,29,30,35,37,40,41,42,43,44,45,47,48,49,50,52,53,55,56,57,59,61,67,68,69,71


source /home/iprnb/venvs/neural-decomposition/bin/activate

module load flexiblas
export FLEXIBLAS=imkl

#Augmented
#_option1_gw3_h64_actor-lr0.01_critic-lr0.01_ent-coef0.03_clip-coef0.3_visit-bonus1_ep-len50-ent_an0


OMP_NUM_THREADS=1 python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --actor_lr 0.01 \
    --critic-lr 0.01 \
    --max_episode_length 50 \
    --ent_coef 0.03 \
    --clip_coef 0.3 \
    --visitation_bonus 1 \
    --anneal_entropy 0 \
    --use_options 1 \
    --env_seed 12 \
    --env_id "ComboGrid" \
    --game_width 3 \
    --total_timesteps 3000000 \
    --exp_mode "augmented" \
    --save_run_info 1 \
    --processed_options 0
