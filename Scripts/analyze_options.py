from collections import defaultdict
import copy
import multiprocessing as mp
from typing import Counter
mp.set_start_method('spawn', force=True)
import numpy as np


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
        input_path = os.path.join(args.analyze_output_path, "option_data.pkl")
        if os.path.exists(input_path):
            with open(input_path, "rb") as f:
                self.data = pickle.load(f)
        else: 
            self.data = []
        self.cur_option = None
        self.cur_option_step = 0
        self.save_every_n_steps = 25_000
        self.finalized = False
    
    def __call__(self, agent, observation, action, next_observation, reward, terminated, truncated, step):
        if agent.current_high is not None and (agent.current_high >= agent.num_primitives) and agent.option_step == 1:
            opt_idx = agent.current_high - agent.num_primitives
            self.cur_option = agent.options[opt_idx]
            self.cur_option_step = 0
            print(f"initiating option {opt_idx} at step {step}")

        if self.cur_option is not None:
            opt_idx = agent.current_high - agent.num_primitives
            self.data.append({
                "step": step,
                "option": opt_idx,
                "option_length": self.cur_option.max_len,  # <- collect actual option object
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "cur_option_step": self.cur_option_step,
                "observation": copy.deepcopy(observation),
                "next_observation": copy.deepcopy(next_observation),
            })
            self.cur_option_step += 1
            if self.cur_option_step == self.cur_option.max_len or terminated or truncated:
                self.cur_option = None
                self.cur_option_step = 0

        if step % self.save_every_n_steps == 0:
            self.finalize()
        else:
            self.finalized = False

    def finalize(self):
        if not self.finalized:
            if len(self.data) == 0:
                print("No option data collected.")
                return
            
            # Save raw data
            output_path = os.path.join(self.args.analyze_input_path, "option_data.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"Saved option data to {output_path}")

            self.analyze()
        else:
            print("Callback already finalized. Skipping save.")

    def analyze(self):
        # Count how many times each option was used (when initiated)
        option_counts = Counter([entry["option"] for entry in self.data if entry["cur_option_step"] == 1])
        print("\nOption usage counts (when started):")
        for opt, count in sorted(option_counts.items()):
            print(f"Option {opt}: {count} times")

        # --- NEW: Distribution of action sequences per option execution ---
        print("\nDistribution of action sequences per option:")

        option_sequences = defaultdict(Counter)
        cur_sequence = []
        cur_option = None
        cur_len = None

        for entry in self.data:
            if entry["cur_option_step"] == 0:
                # New option execution started
                cur_option = entry["option"]
                cur_len = entry["option_length"]
                cur_sequence = [entry["action"]]
            elif cur_option is not None:
                cur_sequence.append(entry["action"])

            if cur_option is not None and len(cur_sequence) == cur_len:
                option_sequences[cur_option][tuple(cur_sequence)] += 1
                cur_option = None
                cur_sequence = []

        for opt, seq_counter in option_sequences.items():
            print(f"\nOption {opt} action sequences:")
            for seq, count in seq_counter.items():
                print(f"  {list(seq)}: {count} times")

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

        # Frequency of option starts by initial observation
        obs_option_counts = defaultdict(Counter)
        for entry in self.data:
            if entry["cur_option_step"] == 1:
                obs = entry["observation"]
                grid = obs[:self.args.game_width * self.args.game_width].tolist()
                obs_key = str(grid) if isinstance(grid, int) else tuple(grid)
                obs_option_counts[entry["option"]][obs_key] += 1

        print("\nOption start frequencies by observation:")
        for option_idx, opt_count in obs_option_counts.items():
            print(f"Option {option_idx}: {dict(opt_count)}")
            
            heatmap = np.full((self.args.game_width, self.args.game_width), np.nan)
            for one_hot, val in opt_count.items():
                idx = one_hot.index(1)
                row, col = divmod(idx, self.args.game_width)
                heatmap[row, col] = val
            plt.figure(figsize=(4, 4))
            plt.imshow(heatmap, cmap='hot', origin='upper')
            plt.colorbar(label='Value')
            plt.title(f'Option {option_idx} Start Frequencies by Observation')
            plt.xlabel('')
            plt.ylabel('')

            # Save to file
            plt.savefig(os.path.join(self.args.analyze_output_path, f"option_{option_idx}_init_freq_heatmap.png"), dpi=300, bbox_inches='tight')  # or 'heatmap.pdf', 'heatmap.jpg', etc.
            plt.show()

        # --- NEW: Distribution of next_observations after each option completes ---
        next_obs_counts = defaultdict(Counter)
        for entry in self.data:
            if entry["cur_option_step"] == entry["option_length"] - 1:
                grid = entry["next_observation"][:self.args.game_width * self.args.game_width].tolist()
                next_obs_key = str(grid) if isinstance(grid, int) else tuple(grid)
                next_obs_counts[entry["option"]][next_obs_key] += 1
        next_obs_counts = defaultdict(Counter)

        print("\nDistribution of next_observations after each option ends:")
        for opt, counter in next_obs_counts.items():
            print(f"Option {opt}: {dict(counter)}")
            heatmap = np.full((self.args.game_width, self.args.game_width), np.nan)
            for one_hot, val in counter.items():
                idx = one_hot.index(1)
                row, col = divmod(idx, self.args.game_width)
                heatmap[row, col] = val
            plt.figure(figsize=(4, 4))
            plt.imshow(heatmap, cmap='hot', origin='upper')
            plt.colorbar(label='Value')
            plt.title(f'Option {opt} End Frequencies by Observation')
            plt.xlabel('')
            plt.ylabel('')

            # Save to file
            plt.savefig(os.path.join(self.args.analyze_output_path, f"option_{opt}_end_freq_heatmap.png"), dpi=300, bbox_inches='tight')  # or 'heatmap.pdf', 'heatmap.jpg', etc.
            plt.show()

        self.finalized = True


def analyze_options(args):

    if os.path.exists(f"{args.analyze_input_path}/option_data.pkl"):
        callback = CallBack(args)
        callback.analyze()
        return
    else:
        print(f"{args.analyze_input_path}/option_data.pkl", "does not exist.")

        option_dir = os.path.join(args.res_dir, args.option_exp_name)
        file_name = f"selected_options.pt" if args.max_num_options is None else f"selected_options_{args.max_num_options}.pt"
        if not os.path.exists(os.path.join(option_dir, file_name)):
            print("Selected Options Doesn't exists!", os.path.join(option_dir, file_name))
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
        

        if args.exp_options_total_steps > 0 and args.exp_options_total_episodes == 0:
            
            callbacks = [CallBack(args)]

            result, best_agent = agent_environment_step_loop(env, agent, args.exp_options_total_steps, writer=None, callbacks=callbacks, save_frame_freq=args.option_save_frame_freq)
        elif args.exp_options_total_episodes > 0 and args.exp_options_total_steps == 0:
            raise NotImplementedError("Episode loop is not implemented for options yet.")
            result, best_agent = agent_environment_episode_loop(env, agent, args.exp_options_total_episodes, writer=None, save_frame_freq=args.option_save_frame_freq)
        else:
            raise ValueError("Both steps and episodes are greater than 0")
        
        env.close()
        return result