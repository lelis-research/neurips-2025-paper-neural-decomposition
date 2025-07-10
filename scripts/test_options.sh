#!/bin/bash
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-1:00      # time (DD-HH:MM)
#SBATCH --output=%j-%N.out  # %N for node name, %j for jobID
#SBATCH --account=aip-lelis
#SBATCH --array=0-14,16-29

cd /home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition


module load flexiblas
export FLEXIBLAS=blis2

source /home/rezaabdz/scratch/envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

OMP_NUM_THREADS=1 python -m pipelines.test_options --seed=$SLURM_ARRAY_TASK_ID \
    --test_env_seeds=14 \
    --wandb_project_name="NEURIPS_2025_test2" \
    --exp_id="extract_wholeDecOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
    --test_exp_name="test_wholeDecOption" \
    --learning_rate=0.001 \
    --clip_coef=0.5 \
    --ent_coef=0.1 \
    # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
    # --test_exp_name="test_learnOptions_internal_filtered" \
    # --learning_rate=0.001 \
    # --clip_coef=0.4 \
    # --ent_coef=0.05 \
    # --exp_id="extract_fineTuning_notFiltered_ComboGrid_gw5_h64_l10_envsd0,1,2,3_selectTypelocal_search_reg0.0maxNumOptions5" \
    # --test_exp_name="test_fine_tuning_unfiltered" \
    # --learning_rate=0.0005 \
    # --clip_coef=0.2 \
    # --ent_coef=0.05 \
    # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
    # --test_exp_name="test_learnOptions_both_filtered" \
    # --learning_rate=0.005 \
    # --clip_coef=0.1 \
    # --ent_coef=0.1 \
    # --exp_id="extract_basePolicyTransferred_ComboGrid_gw5_h64_envsd0,1,2,3" \
    # --test_exp_name="test_base_policy_transferred" \
    # --learning_rate=0.001 \
    # --clip_coef=0.15 \
    # --ent_coef=0.05 \
    # --method="no_options" \
    # --test_exp_name="test_no_options" \
    # --learning_rate=0.0005 \
    # --clip_coef=0.15 \
    # --ent_coef=0.05 \
    # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinput_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
    # --test_exp_name="test_learnOptions_input_filtered" \
    # --learning_rate=0.0005 \
    # --clip_coef=0.15 \
    # --ent_coef=0.05 \
    
    