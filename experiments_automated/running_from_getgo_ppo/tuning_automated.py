# make_slurm_sweep_envsplit.py
from pathlib import Path
from itertools import product
import subprocess

# =========================
# Fixed, project-wide setup
# =========================
PROJECT_DIR = "/home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition"
PY_ENV_ACTIVATE = "/home/rezaabdz/scratch/envs/venv/bin/activate"
# CONFIG_PATH = "configs/combogrid/config_a2c_train.py"
CONFIG_PATH = "configs/combogrid/config_a2c_option.py"
ACCOUNT = "aip-lelis"

# Slurm resources
SLURM_MINUTES = 60        # total minutes per job
SLURM_CPUS    = 1
SLURM_MEM     = "1G"

# =========================
# Sweep space & constants
# =========================
# Set MODE to "train" or "test_option"
MODE = "train"
SAVE_RESULTS = "False"
ENV_NAME = "ComboGrid"
AGENT_CLASS = "PPOAgent"
VERSION = "v0"

# Model scale
GAME_WIDTHS  = [5]
HIDDEN_SIZES = [64]

# Option settings (used only when MODE == "test_option")
# Each tuple is (TMP_OPT, MASK_TYPE)
TMP_OPT_MASK_TYPE = [
    # ("Transfer", ""),
    # ("DecWhole", ""),
    # ("FineTune", "network"),
    # ("Mask", "network"),
    # ("Mask", "input"),
    # ("Mask", "both"),
    ("Vanilla", ""),
]

# Hyperparameter grid
TOTAL_STEPS_SET = [300_000]
STEP_SIZES      = [0.05, 0.01, 0.005, 0.001, 0.0005]
CLIP_RATIOS     = [0.10, 0.15, 0.20, 0.25, 0.30]
ENTROPY_COEFS   = [0.00, 0.05, 0.10, 0.15, 0.20]

# Seeds
agent_seeds_spec = "0,1,2"          # Slurm array: agent seeds
# env_seeds_list   = [0, 1, 2, 3]   # separate submission per env_seed
env_seeds_list   = [12]   # separate submission per env_seed

# =========================
# Paths (SEPARATED!)
# =========================
# RES_DIR (env) is a *top-level* results root chosen by you:
RESULTS_ROOT = Path("Results_ComboGrid_gw5h64_PPO_ReLU_third_round_vanilla_tuning")

# =========================
# Helpers
# =========================
def slurm_time_from_minutes(total_minutes: int) -> str:
    # returns D-HH:MM:SS
    days = total_minutes // (24 * 60)
    rem  = total_minutes % (24 * 60)
    hours = rem // 60
    mins  = rem % 60
    return f"{days}-{hours:02d}:{mins:02d}:00"

def combo_name(total_steps, step_size, clip, ent):
    # Example: steps300000_lr5e-03_clip0.25_ent0.20
    return f"steps{total_steps}_lr{step_size:.0e}_clip{clip:.2f}_ent{ent:.2f}".replace("+", "")

def tok(s: str) -> str:
    return s.strip().lower().replace(" ", "") or "none"

def write_and_submit_slurm(
    job_dir: Path,
    job_name: str,
    agent_seeds_spec: str,
    env_seed: int,
    env_name: str,
    agent_class: str,
    gw: int,
    hs: int,
    total_steps: int,
    step_size: float,
    clip_ratio: float,
    entropy_coef: float,
    mode: str,
    results_root: Path,
    tmp_opt: str = "",
    mask_type: str = "",
):
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    slurm_time = slurm_time_from_minutes(SLURM_MINUTES)

    extra_env = ""
    if mode == "test_option":
        extra_env = f"""
# Option testing
export TMP_OPT="{tmp_opt}"
export MASK_TYPE="{mask_type}"
"""

    # IMPORTANT: RES_DIR is the *top-level* results root, not the job_dir
    slurm_script = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --time={slurm_time}
#SBATCH --mem={SLURM_MEM}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --account={ACCOUNT}
#SBATCH --array={agent_seeds_spec}
#SBATCH --output={logs_dir}/exp_%A_%a.out
#SBATCH --error={logs_dir}/exp_%A_%a.err

set -euo pipefail

cd {PROJECT_DIR}

module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
source {PY_ENV_ACTIVATE}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=blis2

# Fixed env
export MODE="{mode}"
export SAVE_RESULTS="{SAVE_RESULTS}"
export ENV_NAME="{env_name}"
export AGENT_CLASS="{agent_class}"

# Hyperparams for this combination
export GAME_WIDTH={gw}
export HIDDEN_SIZE={hs}
export TOTAL_STEPS={total_steps}
export STEP_SIZE={step_size}
export CLIP_RATIO={clip_ratio}
export ENTROPY_COEF={entropy_coef}{extra_env}

# RESULTS ROOT (distinct from job_dir)
export RES_DIR="{results_root.as_posix()}"

# Agent seed from Slurm array
export SEED=$SLURM_ARRAY_TASK_ID

# Fixed env seed (separate submission per env_seed)
export ENV_SEED={env_seed}

echo "[RUN] MODE=$MODE SEED=$SEED ENV_SEED=$ENV_SEED GW=$GAME_WIDTH HS=$HIDDEN_SIZE steps=$TOTAL_STEPS lr=$STEP_SIZE clip=$CLIP_RATIO ent=$ENTROPY_COEF TMP_OPT=${{TMP_OPT:-}} MASK_TYPE=${{MASK_TYPE:-}} RES_DIR=$RES_DIR"
python -u main.py --config_path {CONFIG_PATH}
"""
    script_path = job_dir / "job.slurm"
    script_path.write_text(slurm_script)

    print(f"Submitting: {script_path}")
    subprocess.run(["sbatch", str(script_path)], check=True)

# =========================
# Main: generate & submit
# =========================
if __name__ == "__main__":
    for gw, hs in product(GAME_WIDTHS, HIDDEN_SIZES):
        # job tree lives *inside* RESULTS_ROOT, but RES_DIR env stays equal to RESULTS_ROOT
        base_jobs = RESULTS_ROOT / f"combogridgw{gw}h{hs}" / "manual_tuning" / MODE

        for env_seed in env_seeds_list:
            env_base = base_jobs
            env_base.mkdir(parents=True, exist_ok=True)

            common_iter = product(TOTAL_STEPS_SET, STEP_SIZES, CLIP_RATIOS, ENTROPY_COEFS)

            if MODE == "train":
                for total_steps, step_size, clip, ent in common_iter:
                    name = combo_name(total_steps, step_size, clip, ent)
                    job_dir = env_base / VERSION / f"env_sd{env_seed}" / name
                    job_dir.mkdir(parents=True, exist_ok=True)

                    job_name = f"train_gw{gw}_h{hs}_es{env_seed}_{name}"
                    write_and_submit_slurm(
                        job_dir=job_dir,
                        job_name=job_name,
                        agent_seeds_spec=agent_seeds_spec,
                        env_seed=env_seed,
                        env_name=ENV_NAME,
                        agent_class=AGENT_CLASS,
                        gw=gw,
                        hs=hs,
                        total_steps=total_steps,
                        step_size=step_size,
                        clip_ratio=clip,
                        entropy_coef=ent,
                        mode=MODE,
                        results_root=RESULTS_ROOT,  # <= RES_DIR env
                    )

            elif MODE == "test_option":
                for (total_steps, step_size, clip, ent), (tmp_opt, mask_type) in product(common_iter, TMP_OPT_MASK_TYPE):
                    name = combo_name(total_steps, step_size, clip, ent)
                    opt_tok, mask_tok = tok(tmp_opt), tok(mask_type)
                    # Job/output dir path format you requested:
                    # Results_.../combogridgw5h64/manual_tuning/v1/env_sd12/steps..._opt-mask_mask-both
                    job_dir = env_base / f"env_sd{env_seed}" / f"opt-{opt_tok}_mask-{mask_tok}" / VERSION / name
                    job_dir.mkdir(parents=True, exist_ok=True)

                    job_name = f"testopt_gw{gw}_h{hs}_es{env_seed}_{name}_{opt_tok}_{mask_tok}"
                    write_and_submit_slurm(
                        job_dir=job_dir,
                        job_name=job_name,
                        agent_seeds_spec=agent_seeds_spec,
                        env_seed=env_seed,
                        env_name=ENV_NAME,
                        agent_class=AGENT_CLASS,
                        gw=gw,
                        hs=hs,
                        total_steps=total_steps,
                        step_size=step_size,
                        clip_ratio=clip,
                        entropy_coef=ent,
                        mode=MODE,
                        results_root=RESULTS_ROOT,  # <= RES_DIR env
                        tmp_opt=tmp_opt,
                        mask_type=mask_type,
                    )
            else:
                raise ValueError(f"Unsupported MODE: {MODE!r}. Use 'train' or 'test_option'.")
