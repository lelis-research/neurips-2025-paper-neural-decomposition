from collections import defaultdict
from copy import copy
import multiprocessing as mp
from typing import Counter
mp.set_start_method('spawn', force=True)


import os
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from Environments.GetEnvironment import get_env
# from Agents.PPOAgent import PPOAgent
from Agents.A2CAgentOption import A2CAgentOption
from Experiments.EnvAgentLoops import agent_environment_step_loop, agent_environment_episode_loop

class CallBack:

    def __init__(self, args):
        self.args = args
        self.data = []
        self.cur_option = None
        self.cur_option_step = 0
    
    def __call__(self, agent, observation, action, actual_reward, terminated, truncated, step):
        if action >= agent.num_primitives:
            opt_idx = action - agent.num_primitives
            self.cur_option = agent.options[opt_idx]
            self.cur_option_step = 1

        if self.cur_option is not None:
                opt_idx = action - agent.num_primitives
                self.data.append({
                "step": step,
                "option": opt_idx,
                "action": action,
                "reward": actual_reward,
                "terminated": terminated,
                "truncated": truncated,
                "cur_option_step": self.cur_option_step,
                "observation": copy.deepcopy(observation),
            })
        if self.cur_option_step == self.cur_option.max_len:
            self.cur_option = None
            self.cur_option_step = 0

    def finalize(self):
        if len(self.data) == 0:
            print("No option data collected.")
            return

        # Save raw data
        output_path = os.path.join(self.args.analyze_output_path, "option_data.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Saved option data to {output_path}")

        # Count how many times each option was used
        option_counts = Counter([entry["option"] for entry in self.data if entry["cur_option_step"] == 1])
        print("\nOption usage counts (when started):")
        for opt, count in sorted(option_counts.items()):
            print(f"Option {opt}: {count} times")

        # Histogram of option usage over steps
        step_bins = defaultdict(list)
        for entry in self.data:
            if entry["cur_option_step"] == 1:
                step_bins[entry["option"]].append(entry["step"])

        plt.figure()
        for opt in sorted(step_bins):
            plt.hist(step_bins[opt], bins=50, alpha=0.6, label=f"Option {opt}")
        plt.xlabel("Step")
        plt.ylabel("Frequency")
        plt.title("Histogram of Option Usage Over Steps")
        plt.legend()
        plt.tight_layout()
        hist_path = os.path.join(self.args.analyze_output_path, "option_usage_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

        # Frequency of options per initial observation
        obs_option_counts = defaultdict(Counter)
        for entry in self.data:
            if entry["cur_option_step"] == 1:
                obs = entry["observation"]
                obs_key = str(obs) if isinstance(obs, int) else tuple(obs)  # assumes obs is discrete/int or list
                obs_option_counts[obs_key][entry["option"]] += 1

        print("\nOption start frequencies by observation:")
        for obs_key, opt_count in obs_option_counts.items():
            print(f"Observation {obs_key}: {dict(opt_count)}")
            


def analyze_options(args):
    option_dir = os.path.join(args.res_dir, args.option_exp_name)
    file_name = f"selected_options.pt" if args.max_num_options is None else f"selected_options_{args.max_num_options}.pt"
    if not os.path.exists(os.path.join(option_dir, file_name)):
        print("Selected Options Doesn't exists!")
        return None
    best_options = torch.load(os.path.join(option_dir, file_name), weights_only=False)

    print(f"Loaded Options from: {os.path.join(option_dir, file_name)}")
    print("Num options: ", len(best_options))
    
    test_option_dir = f"{option_dir}_{args.test_option_env_name}_{file_name[:-3]}_{args.option_name_tag}"
    env = get_env(env_name=args.test_option_env_name,
                  env_params=args.test_option_env_params,
                  wrapping_lst=args.test_option_env_wrappers,
                  wrapping_params=args.test_option_wrapping_params,
                  render_mode=args.test_option_render_mode,
                  max_steps=args.test_option_env_max_steps
                  )
    print(f"Obs Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # keys = ["gamma", "lamda",
    #             "epochs", "total_steps", "rollout_steps", "num_minibatches",
    #             "flag_anneal_step_size", "step_size",
    #             "entropy_coef", "critic_coef",  "clip_ratio", 
    #             "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
    #             "flag_anneal_var", "var_coef",
    #             ]
    keys = ["gamma", "step_size", "rollout_steps", "lamda"]
    
    agent_kwargs = {k: getattr(args, k) for k in keys}

    agent = A2CAgentOption(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                           env.single_action_space if hasattr(env, "single_action_space") else env.action_space, 
                            best_options,
                            device=args.device,
                            **agent_kwargs
                            )
     
    writer = None
    if args.option_save_results:
        if not os.path.exists(test_option_dir):
            os.makedirs(test_option_dir)
        else:
            raise ValueError(f"Experiment directory {test_option_dir} already exists.")
        writer = SummaryWriter(log_dir=test_option_dir)

    if args.exp_options_total_steps > 0 and args.exp_options_total_episodes == 0:
        
        callbacks = [CallBack()]

        result, best_agent = agent_environment_step_loop(env, agent, args.exp_options_total_steps, writer=writer, callbacks=callbacks, save_frame_freq=args.option_save_frame_freq)
    elif args.exp_options_total_episodes > 0 and args.exp_options_total_steps == 0:
        raise NotImplementedError("Episode loop is not implemented for options yet.")
        result, best_agent = agent_environment_episode_loop(env, agent, args.exp_options_total_episodes, writer=writer, save_frame_freq=args.option_save_frame_freq)
    else:
        raise ValueError("Both steps and episodes are greater than 0")
    

    if args.save_results:
        with open(os.path.join(test_option_dir, "res.pkl"), "wb") as f:
            pickle.dump(result, f)
        agent.save(os.path.join(test_option_dir, "final.pt"))
        best_agent.save(os.path.join(test_option_dir, "best.pt"))
    env.close()
    return result