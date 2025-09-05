import subprocess
import textwrap
import json 

def submit_slurm_job(script_content: str, dependency_job_id: str | None = None) -> str:
    """
    Submits a Slurm job script (read from stdin) and returns the new job ID.
    Uses --parsable for robust ID parsing.
    """
    command = ['sbatch', '--parsable']
    if dependency_job_id:
        command.append(f'--dependency=afterok:{dependency_job_id}')

    # Ensure #SBATCH directives start at column 0
    script = textwrap.dedent(script_content).lstrip('\n')

    print(f"SUBMITTING JOB: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            input=script,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("sbatch failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # sbatch --parsable typically returns "123456" or "123456;queue"
    job_id = result.stdout.strip().split(';')[0]
    print(f"Submitted job. Job ID: {job_id}\n")
    return job_id


def _csv_str(seq) -> str:
    """Quote and comma-join a Python iterable without spaces."""
    return '"' + ",".join(str(x) for x in seq) + '"'


def main():
    env_seeds = {
        "MiniGrid-SimpleCrossingS9N1-v0": (0, 1, 2),
        "MiniGrid-FourRooms-v0": (8, 51),
        "MiniGrid-Unlock-v0": (1, 3, 17),
        "MiniGrid-MultiRoom-v0": (431,),
        "ComboGrid": {
            "train": (0, 1, 2, 3),
            "test": (12,)
        }
    }

    width = 6
    train_env = "ComboGrid"
    test_env = "ComboGrid"
    
    base_train_script = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:30:00
#SBATCH --output=combotrain/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-119

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --env_id "ComboGrid" \
    --num_steps 2000 \
    --game_width {width} \
    --total_timesteps 1000000 \
    --save_run_info 0 \
    --method "no_options" \
    --option_mode "vanilla" \
    --mask_type "both" \
    --sweep_run 1 \
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["train"])}
"""

    didec_reg_sweep_script = f"""#!/bin/bash
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=1G
#SBATCH --time=06:55:00
#SBATCH --output=selecting_options/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-44

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

seeds=(54 208 310)
reg_coefs=(0 0.01 0.05 0.1 0.25)
mask_types=("internal" "input" "both")

num_seed=${{#seeds[@]}}
num_reg_coefs=${{#reg_coefs[@]}}
num_types=${{#mask_types[@]}}

idx=$SLURM_ARRAY_TASK_ID

reg_index=$(( idx % num_reg_coefs ))
idx=$(( idx / num_reg_coefs ))

mask_index=$(( idx % num_types ))
idx=$(( idx / num_types ))

sd_index=$(( idx % num_seed ))

SD="${{seeds[${{sd_index}}]}}"
REG="${{reg_coefs[${{reg_index}}]}}"
MASK="${{mask_types[${{mask_index}}]}}"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
    --seed "${{SD}}" \
    --game_width {width} \
    --cpus "$SLURM_CPUS_PER_TASK" \
    --option_mode "didec" \
    --reg_coef "${{REG}}" \
    --mask_type "${{MASK}}"
"""

    didec_models_sweep_scripts = []
    for mask in ["internal", "input", "both"]:
        for i in range(3):
            low = "0" if i == 0 else "1"
            high = "1000" if i != 2 else "699"
            didec_models_sweep_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:20:00
#SBATCH --output=didec-model-sweep/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array={low}-{high}

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

seeds=(0 2 3)                       # 3
learning_rates=(0.01 0.005 0.001 0.0005 0.00005)   # 5
clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)             # 6
ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)            # 6
reg_coefs=(0 0.01 0.05 0.1 0.25)                   # 5

num_seed=${{#seeds[@]}}
num_lr=${{#learning_rates[@]}}
num_ent_coef=${{#ent_coefs[@]}}
num_clip_coef=${{#clip_coef[@]}}
num_reg=${{#reg_coefs[@]}}

idx=$(( SLURM_ARRAY_TASK_ID + {i * 1000} ))

lr_index=$(( idx % num_lr ));         idx=$(( idx / num_lr ))
ent_index=$(( idx % num_ent_coef ));  idx=$(( idx / num_ent_coef ))
clip_index=$(( idx % num_clip_coef )); idx=$(( idx / num_clip_coef ))
reg_index=$(( idx % num_reg ));       idx=$(( idx / num_reg ))
sd_index=$(( idx % num_seed ))

SD="${{seeds[$sd_index]}}"
LR="${{learning_rates[$lr_index]}}"
ENT="${{ent_coefs[$ent_index]}}"
CLIP="${{clip_coef[$clip_index]}}"
REG="${{reg_coefs[$reg_index]}}"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SD" \\
    --learning_rate "$LR" \\
    --ent_coef "$ENT" \\
    --num_steps 2000 \\
    --clip_coef "$CLIP" \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "options" \\
    --option_mode "didec" \\
    --reg_coef "$REG" \\
    --mask_type "{mask}" \\
    --sweep_run 0 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")

    decwhole_options_train_script = f"""#!/bin/bash
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:20:00
#SBATCH --output=selecting_options_decwhole/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=54,208,310,71,115,433,38,438,473,381,73,103,389,267,463,251,398,171,37,348,459,72,471,421,335,480,435,303

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"


python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --game_width {width} \
    --cpus "$SLURM_CPUS_PER_TASK" \
    --option_mode "dec-whole"
"""

    finetune_options_train_script = f"""#!/bin/bash
#SBATCH --cpus-per-task=55
#SBATCH --mem-per-cpu=1G
#SBATCH --time=06:55:00
#SBATCH --output=selecting_options_finetune/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=54,208,310,71,115,433,38,438,473,381,73,103,389,267,463,251,398,171,37,348,459,72,471,421,335,480,435,303

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"


python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/algorithms/fine_tuning.py \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --game_width {width} \
    --cpus "$SLURM_CPUS_PER_TASK"
"""

    neuralaug_options_train_script = f"""#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --output=selecting_options_neuralaug/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=54,208,310,71,115,433,38,438,473,381,73,103,389,267,463,251,398,171,37,348,459,72,471,421,335,480,435,303

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"


python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/base_policy_transferred.py \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --game_width {width}
"""

    methods_sweep_scripts = []
    for method in ["dec-whole", "neural-augmented", "fine-tune"]:
        methods_sweep_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:20:00
#SBATCH --output={method}-model-sweep/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-539

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

seeds=(0 2 3)                       # 3
learning_rates=(0.01 0.005 0.001 0.0005 0.00005)   # 5
clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)             # 6
ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)            # 6
reg_coefs=(0)                   # 1

num_seed=${{#seeds[@]}}
num_lr=${{#learning_rates[@]}}
num_ent_coef=${{#ent_coefs[@]}}
num_clip_coef=${{#clip_coef[@]}}
num_reg=${{#reg_coefs[@]}}

idx=$(( SLURM_ARRAY_TASK_ID + 0 ))

lr_index=$(( idx % num_lr ));         idx=$(( idx / num_lr ))
ent_index=$(( idx % num_ent_coef ));  idx=$(( idx / num_ent_coef ))
clip_index=$(( idx % num_clip_coef )); idx=$(( idx / num_clip_coef ))
reg_index=$(( idx % num_reg ));       idx=$(( idx / num_reg ))
sd_index=$(( idx % num_seed ))

SD="${{seeds[$sd_index]}}"
LR="${{learning_rates[$lr_index]}}"
ENT="${{ent_coefs[$ent_index]}}"
CLIP="${{clip_coef[$clip_index]}}"
REG="${{reg_coefs[$reg_index]}}"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SD" \\
    --learning_rate "$LR" \\
    --ent_coef "$ENT" \\
    --num_steps 2000 \\
    --clip_coef "$CLIP" \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "options" \\
    --option_mode "{method}" \\
    --reg_coef "$REG" \\
    --mask_type "both" \\
    --sweep_run 0 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")
#         methods_sweep_scripts.append(f"""#!/bin/bash
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=1G
# #SBATCH --time=01:20:00
# #SBATCH --output={method}-model-sweep/%A-%a.out
# #SBATCH --account=aip-lelis
# #SBATCH --array=1-79

# set -euo pipefail

# source /home/iprnb/venvs/neural-decomposition/bin/activate

# export FLEXIBLAS=imkl
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export PYTHONPATH=":$PYTHONPATH"

# seeds=(0 1 4 5 6 7)                       # 3
# learning_rates=(0.01 0.005 0.001 0.0005 0.00005)   # 5
# clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)             # 6
# ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)            # 6
# reg_coefs=(0)                   # 1

# num_seed=${{#seeds[@]}}
# num_lr=${{#learning_rates[@]}}
# num_ent_coef=${{#ent_coefs[@]}}
# num_clip_coef=${{#clip_coef[@]}}
# num_reg=${{#reg_coefs[@]}}

# idx=$(( SLURM_ARRAY_TASK_ID + 1000 ))

# lr_index=$(( idx % num_lr ));         idx=$(( idx / num_lr ))
# ent_index=$(( idx % num_ent_coef ));  idx=$(( idx / num_ent_coef ))
# clip_index=$(( idx % num_clip_coef )); idx=$(( idx / num_clip_coef ))
# reg_index=$(( idx % num_reg ));       idx=$(( idx / num_reg ))
# sd_index=$(( idx % num_seed ))

# SD="${{seeds[$sd_index]}}"
# LR="${{learning_rates[$lr_index]}}"
# ENT="${{ent_coefs[$ent_index]}}"
# CLIP="${{clip_coef[$clip_index]}}"
# REG="${{reg_coefs[$reg_index]}}"

# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
#     --seed "$SD" \\
#     --learning_rate "$LR" \\
#     --ent_coef "$ENT" \\
#     --num_steps 2000 \\
#     --clip_coef "$CLIP" \\
#     --env_id {test_env} \\
#     --game_width {width} \\
#     --total_timesteps 1000000 \\
#     --save_run_info 1 \\
#     --method "options" \\
#     --option_mode "{method}" \\
#     --reg_coef "$REG" \\
#     --mask_type "both" \\
#     --sweep_run 0 \\
#     --env_seeds {_csv_str(env_seeds[test_env])}
# """)
    methods_sweep_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:30:00
#SBATCH --output=vanilla-model-sweep/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0-539

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

seeds=(0 2 3)                       # 3
learning_rates=(0.01 0.005 0.001 0.0005 0.00005)   # 5
clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)             # 6
ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)            # 6
reg_coefs=(0)                   # 1

num_seed=${{#seeds[@]}}
num_lr=${{#learning_rates[@]}}
num_ent_coef=${{#ent_coefs[@]}}
num_clip_coef=${{#clip_coef[@]}}
num_reg=${{#reg_coefs[@]}}

idx=$(( SLURM_ARRAY_TASK_ID + 0 ))

lr_index=$(( idx % num_lr ));         idx=$(( idx / num_lr ))
ent_index=$(( idx % num_ent_coef ));  idx=$(( idx / num_ent_coef ))
clip_index=$(( idx % num_clip_coef )); idx=$(( idx / num_clip_coef ))
reg_index=$(( idx % num_reg ));       idx=$(( idx / num_reg ))
sd_index=$(( idx % num_seed ))

SD="${{seeds[$sd_index]}}"
LR="${{learning_rates[$lr_index]}}"
ENT="${{ent_coefs[$ent_index]}}"
CLIP="${{clip_coef[$clip_index]}}"
REG="${{reg_coefs[$reg_index]}}"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SD" \\
    --learning_rate "$LR" \\
    --ent_coef "$ENT" \\
    --num_steps 2000 \\
    --clip_coef "$CLIP" \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "no_options" \\
    --option_mode "vanilla" \\
    --reg_coef "$REG" \\
    --mask_type "both" \\
    --sweep_run 0 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")
#     methods_sweep_scripts.append(f"""#!/bin/bash
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=1G
# #SBATCH --time=01:30:00
# #SBATCH --output=vanilla-model-sweep/%A-%a.out
# #SBATCH --account=aip-lelis
# #SBATCH --array=1-79

# set -euo pipefail

# source /home/iprnb/venvs/neural-decomposition/bin/activate

# export FLEXIBLAS=imkl
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export PYTHONPATH=":$PYTHONPATH"

# seeds=(0 1 4 5 6 7)                       # 3
# learning_rates=(0.01 0.005 0.001 0.0005 0.00005)   # 5
# clip_coef=(0.01 0.05 0.1 0.15 0.2 0.3)             # 6
# ent_coefs=(0.01 0.02 0.03 0.05 0.1 0.2)            # 6
# reg_coefs=(0)                   # 1

# num_seed=${{#seeds[@]}}
# num_lr=${{#learning_rates[@]}}
# num_ent_coef=${{#ent_coefs[@]}}
# num_clip_coef=${{#clip_coef[@]}}
# num_reg=${{#reg_coefs[@]}}

# idx=$(( SLURM_ARRAY_TASK_ID + 1000 ))

# lr_index=$(( idx % num_lr ));         idx=$(( idx / num_lr ))
# ent_index=$(( idx % num_ent_coef ));  idx=$(( idx / num_ent_coef ))
# clip_index=$(( idx % num_clip_coef )); idx=$(( idx / num_clip_coef ))
# reg_index=$(( idx % num_reg ));       idx=$(( idx / num_reg ))
# sd_index=$(( idx % num_seed ))

# SD="${{seeds[$sd_index]}}"
# LR="${{learning_rates[$lr_index]}}"
# ENT="${{ent_coefs[$ent_index]}}"
# CLIP="${{clip_coef[$clip_index]}}"
# REG="${{reg_coefs[$reg_index]}}"

# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
#     --seed "$SD" \\
#     --learning_rate "$LR" \\
#     --ent_coef "$ENT" \\
#     --num_steps 2000 \\
#     --clip_coef "$CLIP" \\
#     --env_id {test_env} \\
#     --game_width {width} \\
#     --total_timesteps 1000000 \\
#     --save_run_info 1 \\
#     --method "no_options" \\
#     --option_mode "vanilla" \\
#     --reg_coef "$REG" \\
#     --mask_type "both" \\
#     --sweep_run 0 \\
#     --env_seeds {_csv_str(env_seeds[test_env])}
# """)


    # base_train_jobid = submit_slurm_job(base_train_script)
    # didec_reg_sweep_jobid = submit_slurm_job(didec_reg_sweep_script)
    # decwhole_option_train_jobid = submit_slurm_job(decwhole_options_train_script)
    # finetune_option_train_jobid = submit_slurm_job(finetune_options_train_script)
    # neuralaug_option_train_jobid = submit_slurm_job(neuralaug_options_train_script)
    # for script in methods_sweep_scripts:
    #     print(submit_slurm_job(script))
    # for script in didec_models_sweep_scripts:
    #     print(submit_slurm_job(script))

    # best_config_per_combo = analyze_auc(
    #     settings=["internal", "input", "both"],
    #     option_mode="didec",
    #     config_filepath=f"binary/configs/ComboGrid_gw{width}.json"
    # )
#     didec_option_all_seeds_scripts = []
#     with open(f"binary/configs/ComboGrid_gw6_withwalls.json", "r") as f:
#             hyperparams = json.load(f)
#             for method in hyperparams:
#                 if "didec" in method:
#                     mask_type = method.split("_")[1]
#                     reg_coef = float(hyperparams[method]["reg_coef"])
#                     didec_option_all_seeds_scripts.append(f"""#!/bin/bash
# #SBATCH --cpus-per-task=30
# #SBATCH --mem-per-cpu=1G
# #SBATCH --time=01:55:00
# #SBATCH --output=selecting_options/%A-%a.out
# #SBATCH --account=aip-lelis
# #SBATCH --array=0,2,3,4,6,7,8,11,14,16,19,21,25,27,28,29,30,33,34,35,36,40,42,43,44,45,49,51,52,53

# set -euo pipefail

# source /home/iprnb/venvs/neural-decomposition/bin/activate

# export FLEXIBLAS=imkl
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export PYTHONPATH=":$PYTHONPATH"

# python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/option_discovery.py \
#     --seed "$SLURM_ARRAY_TASK_ID" \
#     --game_width 6 \
#     --cpus "$SLURM_CPUS_PER_TASK" \
#     --option_mode "didec" \
#     --reg_coef {reg_coef} \
#     --mask_type {mask_type}
# """)

#     for script in didec_option_all_seeds_scripts:
#         submit_slurm_job(script)

    methods_test_scripts = []
    for method in ['fine-tune', 'dec-whole', 'neural-augmented']:
        methods_test_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:20:00
#SBATCH --output={method}-model-test/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0,2,3,4,6,7,8,11,14,16,19,21,25,27,28,29,30,33,34,35,36,40,42,43,44,45,49,51,52,53

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"




python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SLURM_ARRAY_TASK_ID" \\
    --num_steps 2000 \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "options" \\
    --option_mode "{method}" \\
    --mask_type "both" \\
    --sweep_run 1 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")
    methods_test_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=vanilla-model-test/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0,2,3,4,6,7,8,11,14,16,19,21,25,27,28,29,30,33,34,35,36,40,42,43,44,45,49,51,52,53
set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SLURM_ARRAY_TASK_ID" \\
    --num_steps 2000 \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "no_options" \\
    --option_mode "vanilla" \\
    --mask_type "both" \\
    --sweep_run 1 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")
    for script in methods_test_scripts:
        submit_slurm_job(script)


    didec_option_test_problems_scripts = []
    for mask_type in ["both", "internal", "input"]:
        didec_option_test_problems_scripts.append(f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:20:00
#SBATCH --output=didec-test/%A-%a.out
#SBATCH --account=aip-lelis
#SBATCH --array=0,2,3,4,6,7,8,11,14,16,19,21,25,27,28,29,30,33,34,35,36,40,42,43,44,45,49,51,52,53

set -euo pipefail

source /home/iprnb/venvs/neural-decomposition/bin/activate

export FLEXIBLAS=imkl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=":$PYTHONPATH"

python3.11 ~/scratch/neurips-2025-paper-neural-decomposition/pipelines/train_ppo.py \\
    --seed "$SLURM_ARRAY_TASK_ID" \\
    --num_steps 2000 \\
    --env_id {test_env} \\
    --game_width {width} \\
    --total_timesteps 1000000 \\
    --save_run_info 1 \\
    --method "options" \\
    --option_mode "didec" \\
    --mask_type "{mask_type}" \\
    --sweep_run 1 \\
    --env_seeds {_csv_str(env_seeds["ComboGrid"]["test"])}
""")

    for script in didec_option_test_problems_scripts:
        submit_slurm_job(script)


if __name__ == "__main__":
    main()
