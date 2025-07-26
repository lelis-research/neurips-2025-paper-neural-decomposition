import os
import copy
import json
import joblib
import numpy as np
import optuna
from optuna.samplers import GridSampler
from torch.utils.tensorboard import SummaryWriter
import torch

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
    Runs Optuna to tune PPO hyperparams either with:
      - exhaustive grid search (args.exhaustive_search=True), or
      - regular search (TPE by default).
    """
    #initialize environment
    env = get_env(env_name=args.tuning_env_name,
                env_params=args.tuning_env_params,
                wrapping_lst=args.tuning_env_wrappers,
                wrapping_params=args.tuning_wrapping_params,
                render_mode=None,
                max_steps=args.tuning_env_max_steps
                )
    
    # Load options if any
    options_lst = []
    for option_file in args.option_path_tuning:
        option_file = os.path.join(args.res_dir, option_file)
        best_options = torch.load(option_file, weights_only=False)
        options_lst.append(best_options)
    
    
    # 0) prepare results directory
    exp_dir = os.path.join(
        args.res_dir, f"tuning_{args.tuning_env_name}_{args.tuning_nametag}"
    )
    args.res_dir = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        print(f"Tuning Experiment Already Exists at {exp_dir}")
    
    study_kwargs = dict(
        study_name=f"tuning_{args.tuning_env_name}_{args.tuning_nametag}",
        storage=args.tuning_storage
        
        # load_if_exists=True,
        # direction="minimize"
    )

    # 1) If exhaustive, build the grid_space and sampler
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
        except: 
            study = optuna.create_study(**study_kwargs, sampler=sampler)
        # optimize_kwargs = {"n_jobs":args.num_tuning_workers}  # omit n_trials → exhausts the grid
        total_trials = int(np.prod([len(v) for v in grid_space.values()]))

        print(f"Exhaustive search with {total_trials} trials over {len(grid_space)} parameters.")
    else:
        try:
            study = optuna.load_study(**study_kwargs)
        except: 
            study = optuna.create_study(**study_kwargs)
        # optimize_kwargs = {"n_trials": args.num_trials, "n_jobs":args.num_tuning_workers}
        total_trials = args.num_trials
        
    if args.n_trials_per_job is not None:
        n_trials = args.n_trials_per_job
    else:
        if args.exhaustive_search:
            n_trials = None   # GridSampler will exhaustively run all
        else:
            n_trials = args.num1trials
    
    optimize_kwargs = {"n_trials": n_trials, "n_jobs": 1, "callbacks":[ProgressCallBack()], "show_progress_bar":True}

    # 2) The objective always uses trial.suggest_*
    def objective(trial):
        current = trial.number + 1
        print(f"[{current:>3}/{total_trials}] trial with sampler args…")
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

        if args.agent_class == "PPOAgent":
            keys = ["gamma", "lamda",
                        "epochs", "total_steps", "rollout_steps", "num_minibatches",
                        "flag_anneal_step_size", "step_size",
                        "entropy_coef", "critic_coef",  "clip_ratio", 
                        "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
                        "flag_anneal_var", "var_coef", "l1_lambda", # NOTE: l1_lambda wasn't part of this list
                        ]
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
        if len(options_lst) > 0:
            sum_return = 0
            for option in options_lst: 
                if args.agent_class == "A2CAgentOption":
                    agent = agent_class(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                                env.single_action_space if hasattr(env, "single_action_space") else env.action_space, 
                                option,
                                device=args.device,
                                **agent_kwargs
                                )
                else:
                    agent = agent_class(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                                        option,
                                        device=args.device,
                                        **agent_kwargs
                                        )
                    # agent = PPOAgentOption(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                    #                     option,
                    #                     device=args.device,
                    #                     **agent_kwargs
                    #                     )
                result, _ = agent_environment_step_loop(env, agent, args.steps_per_trial, verbose=True)
                sum_return += sum(r['episode_return'] for r in result)
            avg_return = sum_return / len(options_lst)
            
        else:
            sum_return = 0
            assert len(args.seeds) == 1, "Tuning should be done with a single seed"
            SEED = args.seeds[0]
            for seed in args.tuning_seeds:
                set_seed(SEED + seed)
                agent = agent_class(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space,
                        env.single_action_space if hasattr(env, "single_action_space") else env.action_space,
                        device=args.device,
                        **agent_kwargs
                        )
                # agent = PPOAgent(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                #                  env.single_action_space if hasattr(env, "single_action_space") else env.action_space,
                #                 device=args.device,
                #                 **agent_kwargs
                #                 )
                result, _ = agent_environment_step_loop(env, agent, args.steps_per_trial, verbose=True)
                sum_return += sum(r['episode_return'] for r in result)
            avg_return = sum_return / len(args.tuning_seeds)
            set_seed(SEED)
            

        # minimize → negative return -> maximize return
        return -avg_return

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