import os
import copy
import json
import joblib
import numpy as np
import optuna
from optuna.samplers import GridSampler
from torch.utils.tensorboard import SummaryWriter

from Environments.GetEnvironment import get_env
from Experiments.EnvAgentLoops import (
    agent_environment_step_loop,
    agent_environment_episode_loop,
)
from .train_agent import train_parallel_seeds

def tune_ppo(args):
    """
    Runs Optuna to tune PPO hyperparams either with:
      - exhaustive grid search (args.exhaustive_search=True), or
      - regular search (TPE by default).
    """
    # 0) prepare results directory
    exp_dir = os.path.join(
        args.res_dir, f"tuning_{args.tuning_env_name}_{args.steps_per_trial}"
    )
    args.res_dir = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        raise ValueError(f"Tuning Experiment Already Exists at {exp_dir}")

    # 1) If exhaustive, build the grid_space and sampler
    if args.exhaustive_search:
        num_points = getattr(args, "num_grid_points", 5)
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
        study = optuna.create_study(direction="minimize", sampler=sampler)
        optimize_kwargs = {}  # omit n_trials → exhausts the grid
    else:
        study = optuna.create_study(direction="minimize")
        optimize_kwargs = {"n_trials": args.num_trials}

    # 2) The objective always uses trial.suggest_*
    def objective(trial):
        # sample into this dict regardless of mode
        sampled = {}
        if args.exhaustive_search:
            # grid search → categorical over each list in grid_space
            for name, choices in grid_space.items():
                sampled[name] = trial.suggest_categorical(name, choices)
        else:
            # regular search → float/int for tuples, categorical for lists
            for name, bounds in args.param_ranges.items():
                if isinstance(bounds, list):
                    sampled[name] = trial.suggest_categorical(name, bounds)
                else:
                    low, high = bounds
                    if isinstance(low, float) or isinstance(high, float):
                        sampled[name] = trial.suggest_float(name, low, high)
                    else:
                        sampled[name] = trial.suggest_int(name, low, high)

        # build trial_args
        trial_args = copy.deepcopy(args)
        for k, v in sampled.items():
            if not hasattr(trial_args, k):
                raise ValueError(f"agent doesn't have '{k}' argument")
            setattr(trial_args, k, v)

        # fixed experiment settings
        trial_args.exp_total_steps     = args.steps_per_trial
        trial_args.exp_total_episodes  = 0
        trial_args.save_results        = True
        trial_args.training_env_name   = args.tuning_env_name
        trial_args.training_env_params = args.tuning_env_params
        trial_args.training_env_wrappers = args.tuning_env_wrappers
        trial_args.training_wrapping_params = args.tuning_wrapping_params
        trial_args.training_render_mode = None
        trial_args.load_agent = None
        trial_args.nametag   = f"trial_{trial.number}"

        # train & evaluate
        _, best_agents = train_parallel_seeds(trial_args.tuning_seeds, trial_args)
        test_env = get_env(
            env_name=args.tuning_env_name,
            env_params=args.tuning_env_params,
            wrapping_lst=args.tuning_env_wrappers,
            wrapping_params=args.tuning_wrapping_params,
            render_mode=None,
        )
        num_success = 0
        for agent in best_agents:
            result, _ = agent_environment_episode_loop(
                test_env,
                agent,
                total_episodes=10,
                training=False,
                writer=None,
                save_frame_freq=1,
                greedy=True,
            )
            num_success += sum(r["episode_return"] > 0 for r in result)

        # minimize → negative success count
        return -num_success

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