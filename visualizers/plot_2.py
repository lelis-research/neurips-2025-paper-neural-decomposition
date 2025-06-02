import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import imageio
import os
import torch
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_id: str = "FourRooms"

if __name__ == "__main__":
    args = tyro.cli(Args)
        # create shared figure/axes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.tight_layout(pad=3.0)
    # Plotting the mean returns and the 95% CI
    plt.figure(figsize=(10, 6))

    # automatic distinct colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    method_patterns = {
            'D2O': 'FourRooms_dec',
            'Vanilla':'FourRooms_vanilla',
            'dec whole': 'FourRooms_dec-whole',
            'fine tune': 'FourRooms_fine_tune',
            'Neural Augmented': 'FourRooms_augmented'

        }

    for s in [8,41,51]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.tight_layout(pad=3.0)
        # Plotting the mean returns and the 95% CI
        plt.figure(figsize=(10, 6))
        for (method_name, pattern), color in zip(method_patterns.items(), colors):
            returns_all=[]
            steps_all=[]
            print(f"Loading {method_name}")
            # find all matching experiment folders
            ep_returns = []
            runs = []
            for root, _, files in os.walk(f"binary/models/{pattern}"):
                for file in files:
                    if file.endswith('.pt') and file.strip().split("/")[-1].split("-")[1] == str(s):
                        # pytorch_models.append(os.path.join(root, file))
                        model = os.path.join(root, file)
                        seed = int(model[model.find("seed"):][len("seed")+1:model[model.find("seed"):].find(os.sep)])
                        # if seed not in [1,2]: continue
                        checkpoint = torch.load(model, weights_only=False)
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            print(model)
                            state_dict = checkpoint['state_dict']
                            episode_lengths = checkpoint['episode_lengths']
                            returns = checkpoint['average_returns']
                            steps = checkpoint['steps']
                            # runs.append({'episode_return': returns, 'episode_length': steps})

                            returns_all.append(returns)
                            steps_all = steps
                            # print(returns)
                        else:
                            state_dict = checkpoint


            # Convert returns_all into a numpy array for easier manipulation
            returns_all = np.array(returns_all)


            # Calculate the mean and 95% confidence interval for each step
            mean_returns = np.mean(returns_all, axis=0)
            std_returns = np.std(returns_all, axis=0)
            n_samples = len(returns_all)

            # 95% Confidence Interval using z-value for 95% CI (Z ≈ 1.96)
            z_value = 1.96
            ci_lower = mean_returns - z_value * (std_returns / np.sqrt(n_samples))
            ci_upper = mean_returns + z_value * (std_returns / np.sqrt(n_samples))

            # Plot the mean return
            plt.plot(steps_all, mean_returns, label=method_name, color=color, marker='o')

            # Plot the 95% Confidence Interval
            plt.fill_between(steps_all, ci_lower, ci_upper, color=color, alpha=0.3)

        plt.title(f'FourRooms seed {s}')
        plt.xlabel('Steps')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"FourRooms-{s}.png")