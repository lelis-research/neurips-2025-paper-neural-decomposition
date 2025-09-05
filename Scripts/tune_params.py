import os
import copy
import json
import joblib
import numpy as np
import optuna
from optuna.samplers import GridSampler
from torch.utils.tensorboard import SummaryWriter
import torch
import itertools
from optuna.storages import JournalStorage, JournalFileStorage


from Environments.GetEnvironment import get_env
from Experiments.EnvAgentLoops import (
    agent_environment_step_loop,
    agent_environment_episode_loop,
)
from .train_agent import train_parallel_seeds
from Agents.PPOAgentOption import PPOAgentOption
from Agents.PPOAgent import PPOAgent
from Agents.ElitePPOAgent import ElitePPOAgent
from Agents.RandomAgent import RandomAgent
from Agents.SACAgent import SACAgent
from Agents.DDPGAgent import DDPGAgent
from Agents.DQNAgent import DQNAgent
from Agents.NStepDQNAgent import NStepDQNAgent
from Agents.A2CAgent import A2CAgent
from Agents.A2CAgentOption import A2CAgentOption
import random
import warnings


class ProgressCallBack:
    def __init__(self, warn_every_n: int = 1):
        self.warn_every_n = warn_every_n

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            n_completed = len(completed_trials)

            if n_completed % self.warn_every_n == 0:
                warnings.warn(
                    f"Progress: {n_completed} trials completed out of {len(study.trials)} total."
                )


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def tune_agent(args):
    """
    Runs Optuna to tune hyperparams either with:
      - exhaustive grid search (args.exhaustive_search=True), or
      - regular search (TPE by default).

    If args.tuning_parallel_method == "job-based" and exhaustive_search=True,
    this function will run ONLY the single hyperparameter combination indexed
    by args.tuning_job_idx (0-based) from the full grid.
    """
    # Load options if any
    options_lst = []
    for option_file in args.option_path_tuning:
        option_file = os.path.join(args.res_dir, option_file)
        best_options = torch.load(option_file, weights_only=False)
        options_lst.append(best_options)

    # 0) prepare results directory
    exp_dir = os.path.join(args.res_dir, f"tuning_{args.tuning_env_name}_{args.tuning_nametag}")
    args.res_dir = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        print(f"Tuning Experiment Already Exists at {exp_dir}")

    study_kwargs = dict(
        study_name=f"tuning_{args.tuning_env_name}_{args.tuning_nametag}",
        storage=JournalStorage(JournalFileStorage(args.tuning_storage)),
        # direction="minimize"  # rely on default or add if you prefer
    )

    job_based = (getattr(args, "tuning_parallel_method", None) == "job-based")
    assert not (len(options_lst) > 0 and len(args.tuning_seeds) > 0), "Cannot tune over both options and seeds simultaneously."
    num_of_runs_per_trial = len(args.tuning_seeds) if len(args.tuning_seeds) > 0 else len(options_lst)
    assert num_of_runs_per_trial > 0, "Must tune over at least one option or one seed."
    

    # 1) If exhaustive, build the grid_space and sampler
    grid_space = None
    if args.exhaustive_search:
        num_points = getattr(args, "num_grid_points", 10)
        grid_space = {}
        for name, bounds in args.param_ranges.items():
            if isinstance(bounds, list):
                grid_space[name] = bounds
            else:
                low, high = bounds
                if isinstance(low, float) or isinstance(high, float):
                    grid_space[name] = np.linspace(low, high, num_points).tolist()
                else:
                    pts = np.linspace(low, high, num_points)
                    grid_space[name] = [int(round(x)) for x in pts]

        sampler = GridSampler(grid_space)
        try:
            study = optuna.load_study(**study_kwargs, sampler=sampler)
        except Exception:
            study_kwargs['load_if_exists'] = True
            study = optuna.create_study(**study_kwargs, sampler=sampler)

        # total #points in full grid
        total_trials = int(np.prod([len(v) for v in grid_space.values()]))

        if job_based:
            # Build deterministic ordering of the grid and enqueue the requested point
            keys_in_order = list(grid_space.keys())  # preserve insertion order
            grid_lists = [grid_space[k] for k in keys_in_order]

            idx = int(args.tuning_job_idx) // num_of_runs_per_trial
            if idx < 0 or idx >= total_trials:
                raise IndexError(
                    f"tuning_job_idx={idx} is out of range for total grid size {total_trials}."
                )
            combo = list(itertools.product(*grid_lists))[idx]
            chosen_params = {k: combo[i] for i, k in enumerate(keys_in_order)}
            print(f"[JOB-BASED] Running ONLY grid index {idx}/{total_trials-1}: {chosen_params}")

            # Enqueue exactly one trial with these parameters
            study.enqueue_trial(chosen_params)

            # Force exactly one trial
            n_trials = 1
            print(f"Exhaustive search (job-based single point). Total grid={total_trials}.")
        else:
            # normal exhaustive run: let GridSampler exhaust the grid (or per-job n_trials)
            print(f"Exhaustive search with {total_trials} trials over {len(grid_space)} parameters.")
            if getattr(args, "n_trials_per_job", None) is not None:
                n_trials = args.n_trials_per_job
            else:
                n_trials = None  # GridSampler will exhaustively run all

    else:
        # Non-exhaustive (TPE or other sampler)
        if job_based:
            raise ValueError(
                "tuning_parallel_method='job-based' is only supported with exhaustive_search=True."
            )
        try:
            study = optuna.load_study(**study_kwargs)
        except Exception:
            study_kwargs['load_if_exists'] = True
            study = optuna.create_study(**study_kwargs)

        total_trials = args.num_trials
        if getattr(args, "n_trials_per_job", None) is not None:
            n_trials = args.n_trials_per_job
        else:
            n_trials = args.num1trials  # keep user's original arg name

    optimize_kwargs = {
        "n_trials": n_trials,
        "n_jobs": 1,
        "callbacks": [ProgressCallBack()],
        "show_progress_bar": True
    }

    # 2) The objective always uses trial.suggest_*
    def objective(trial):
        # For exhaustive + job-based, Optuna will feed back the enqueued params.
        # We still call suggest_* so the params are recorded consistently.
        if args.exhaustive_search:
            # grid search → categorical over each list in grid_space
            sampled = {}
            for name, choices in grid_space.items():
                sampled[name] = trial.suggest_categorical(name, choices)
            current = trial.number + 1
            print(f"[{current:>3}/{'1' if job_based else str(total_trials)}] trial with grid params…")
        else:
            # regular search → float/int for tuples, categorical for lists
            sampled = {}
            for name, bounds in args.param_ranges.items():
                if isinstance(bounds, list):
                    sampled[name] = trial.suggest_categorical(name, bounds)
                else:
                    low, high = bounds
                    if isinstance(low, float) or isinstance(high, float):
                        sampled[name] = trial.suggest_float(name, low, high)
                    else:
                        sampled[name] = trial.suggest_int(name, low, high)
            current = trial.number + 1
            print(f"[{current:>3}/{total_trials}] trial with sampler args…")

        # build trial_args
        trial_args = copy.deepcopy(args)
        for k, v in sampled.items():
            if not hasattr(trial_args, k):
                raise ValueError(f"agent doesn't have '{k}' argument")
            setattr(trial_args, k, v)

        if args.agent_class == "PPOAgent":
            keys = ["gamma", "lamda",
                    "epochs", "total_steps", "rollout_steps", "num_minibatches",
                    "flag_anneal_step_size", "step_size",
                    "entropy_coef", "critic_coef", "clip_ratio",
                    "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
                    "flag_anneal_var", "var_coef", "l1_lambda"]
        elif args.agent_class in ["DQNAgent", "NStepDQNAgent"]:
            keys = ["gamma", "step_size",
                    "batch_size", "target_update_freq",
                    "epsilon", "replay_buffer_cap",
                    "action_res"]
        elif args.agent_class == "DDPGAgent":
            keys = ["gamma", "tau",
                    "actor_lr", "critic_lr",
                    "buf_size", "batch_size",
                    "noise_phi", "ou_theta", "ou_sigma",
                    "epsilon_end", "decay_steps"]
        elif args.agent_class in ["A2CAgent", "A2CAgentOption"]:
            keys = ["gamma", "step_size", "rollout_steps", "lamda", "hidden_size"]
        else:
            raise NotImplementedError("Agent class not known")

        agent_class = eval(args.agent_class)
        agent_kwargs = {k: getattr(trial_args, k) for k in keys}

        # 
        import concurrent.futures as cf
        
        def _make_env():
            return get_env(
                env_name=args.tuning_env_name,
                env_params=args.tuning_env_params,
                wrapping_lst=args.tuning_env_wrappers,
                wrapping_params=args.tuning_wrapping_params,
                render_mode=None,
                max_steps=args.tuning_env_max_steps
            )

        def _run_with_option(option):
            # fresh env per worker
            _env = _make_env()
            try:
                if args.agent_class == "A2CAgentOption":
                    _agent = agent_class(
                        _env.single_observation_space if hasattr(_env, "single_observation_space") else _env.observation_space,
                        _env.single_action_space if hasattr(_env, "single_action_space") else _env.action_space,
                        option,
                        device=args.device,
                        **agent_kwargs
                    )
                else:
                    _agent = agent_class(
                        _env.single_observation_space if hasattr(_env, "single_observation_space") else _env.observation_space,
                        option,
                        device=args.device,
                        **agent_kwargs
                    )
                result, _ = agent_environment_step_loop(_env, _agent, args.steps_per_trial, verbose=True)
                return sum(r["episode_return"] for r in result)
            finally:
                if hasattr(_env, "close"):
                    _env.close()

        def _run_with_seed(seed):
            _env = _make_env()
            try:
                set_seed(312 + seed)
                _agent = agent_class(
                    _env.single_observation_space if hasattr(_env, "single_observation_space") else _env.observation_space,
                    _env.single_action_space if hasattr(_env, "single_action_space") else _env.action_space,
                    device=args.device,
                    **agent_kwargs
                )
                result, _ = agent_environment_step_loop(_env, _agent, args.steps_per_trial, verbose=True)
                return sum(r["episode_return"] for r in result)
            finally:
                if hasattr(_env, "close"):
                    _env.close()

        # choose parallelism level
        max_workers = int(getattr(args, "num_tuning_workers", 4))
        if max_workers <= 0:
            max_workers = 1

        run_idx = int(args.tuning_job_idx) % num_of_runs_per_trial if job_based else 0
        if len(options_lst) > 0:
            option = options_lst[run_idx]
            print(f"  → running option index {run_idx}/{num_of_runs_per_trial-1}")
            results = _run_with_option(option)  # warmup
            return -results
        else:
            seed = args.tuning_seeds[run_idx]
            print(f"  → running seed index {run_idx}/{num_of_runs_per_trial-1}: {seed}")
            results = _run_with_seed(seed)  # warmup
            return -results

        # if len(options_lst) > 0:
        #     # parallel over options
        #     with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        #         futures = [ex.submit(_run_with_option, opt) for opt in options_lst]
        #         returns = [f.result() for f in futures]
        #     avg_return = float(sum(returns)) / len(options_lst)
        # else:
        #     # parallel over seeds
        #     SEED = 312  # or your base seed
        #     with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        #         futures = [ex.submit(_run_with_seed, seed) for seed in args.tuning_seeds]
        #         returns = [f.result() for f in futures]
        #     avg_return = float(sum(returns)) / len(args.tuning_seeds)
        #     set_seed(SEED)

        # minimize → negative return -> maximize return
        # return -avg_return

    # 3) run optimization
    study.optimize(objective, **optimize_kwargs)

    # 4) save & report
    best_params = study.best_trial.params
    with open(os.path.join(exp_dir, "best_hyperparams.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    joblib.dump(study, os.path.join(exp_dir, "optuna_study.pkl"))

    print(f"Best hyperparameters saved to {exp_dir}/best_hyperparams.json")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, study