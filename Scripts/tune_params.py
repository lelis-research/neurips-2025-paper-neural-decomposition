import numpy as np
import optuna
import copy
import os
import json
import joblib 

from Environments.GetEnvironment import get_env
from .train_agent import train_parallel_seeds

def tune_ppo(args):
    """
    Runs Optuna to tune PPO hyperparams by running parallel seeds
    with a fixed number of steps (step‑loop).
    """
    exp_dir = os.path.join(args.res_dir, f"tuning_{args.tuning_env_name}_{args.steps_per_trial}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        raise ValueError(f"Tuning Experiment Already Exists at {exp_dir}")

    def objective(trial):
        # 1) sample hyperparameters
        sampled = {}
        for k, (low, high) in args.param_ranges.items():
            default = getattr(args, k)
            if isinstance(default, float):
                sampled[k] = trial.suggest_float(k, low, high)
            else:
                sampled[k] = trial.suggest_int(k, low, high)
        
        # 2) build a fresh args for this trial
        trial_args = copy.deepcopy(args)            # start from your dataclass defaults
        for k, v in sampled.items():
            setattr(trial_args, k, v)
       
        trial_args.exp_total_steps = args.steps_per_trial
        trial_args.exp_total_episodes = 0
        trial_args.save_results = False
        trial_args.training_env_name = args.tuning_env_name
        trial_args.training_env_params = args.tuning_env_params
        trial_args.training_env_wrappers = args.tuning_env_wrappers
        trial_args.training_wrapping_params = args.tuning_wrapping_params


        # 3) run train_parallel_seeds
        results = train_parallel_seeds(trial_args.seeds, trial_args)
        # all_metrics is a list (len = #seeds) of lists of episode‑dicts

        # 4) aggregate returns across seeds & episodes
        avg_returns = []
        for r in range(len(results)):
            avg_returns.append([0])
            # each seed_metrics is a list of episode‑dicts
            for ep in results[r]:
                avg_returns[r] += ep["episode_return"] / len(results[r])

        worst_return = min(avg_returns)
        # optuna minimizes → return negative average reward
        return -worst_return

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.num_trials)

    # pull out the best params
    best_params = study.best_trial.params


    # 5) save best params as JSON
    json_path = os.path.join(exp_dir, "best_hyperparams.json")
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to {json_path}")

    # 6) optionally, save the entire Optuna study for later analysis
    study_path = os.path.join(exp_dir, "optuna_study.pkl")
    joblib.dump(study, study_path)
    print(f"Full Optuna study saved to {study_path}")

    # 7) print to console
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, study