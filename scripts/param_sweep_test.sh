#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-0:20
#SBATCH --output=%j-%N.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-719  # 240 experiments × 3 seeds = 720 jobs

module load flexiblas
export FLEXIBLAS=blis2

cd /home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition

source /home/rezaabdz/scratch/envs/venv/bin/activate

# Parameter ranges
learning_rates=(0.0005 0.001 0.005 0.01 0.05)      # 5
clip_coefs=(0.1 0.15 0.2 0.3 0.4 0.5)              # 6
ent_coefs=(0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35)   # 8

num_seeds=3
total_experiments=$((5 * 6 * 8))  # 240

global_idx=$SLURM_ARRAY_TASK_ID
exp_idx=$((global_idx / num_seeds))  # 0–239
seed=$((global_idx % num_seeds))     # 0–2

lr_idx=$((exp_idx / 48))             # 0–4
cc_idx=$(((exp_idx % 48) / 8))       # 0–5
ec_idx=$((exp_idx % 8))              # 0–7

lr=${learning_rates[$lr_idx]}
clip=${clip_coefs[$cc_idx]}
ent=${ent_coefs[$ec_idx]}

echo "Running job $global_idx (exp=$exp_idx, seed=$seed) with:"
echo "  Learning rate: $lr"
echo "  Clip coef: $clip"
echo "  Entropy coef: $ent"
echo "  Seed: $seed"

OMP_NUM_THREADS=1 python -m pipelines.parameter_sweep_testing \
  --seed=$seed \
  --learning_rate=$lr \
  --clip_coef=$clip \
  --ent_coef=$ent \
  --exp_id="extract_wholeDecOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
  --test_exp_name="test_wholeDecOption"
  # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
  # --test_exp_name="test_learnOptions_internal_filtered"
  # --exp_id="extract_fineTuning_notFiltered_ComboGrid_gw5_h64_l10_envsd0,1,2,3_selectTypelocal_search_reg0.0maxNumOptions5" \
  # --test_exp_name="test_fine_tuning_unfiltered"
  # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
  # --test_exp_name="test_learnOptions_both_filtered"
  # --test_exp_name="test_no_options" \
  # --method="no_options" 
  # --exp_id="extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinput_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
  # --test_exp_name="test_learnOptions_input_filtered"

  
  # --exp_id="extract_wholeDecOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5" \
  # --test_exp_name="test_wholeDecOption"
  # --exp_id="extract_fineTuning_notFiltered_ComboGrid_gw5_h64_l10_envsd0,1,2,3_selectTypelocal_search_reg0.0maxNumOptions5" \
  # --test_exp_name="test_fine_tuning_unfiltered"
  # --exp_id="extract_basePolicyTransferred_ComboGrid_gw5_h64_envsd0,1,2,3" \
  # --test_exp_name="test_base_policy_transferred"
  
