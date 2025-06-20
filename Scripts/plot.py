import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import imageio
import os
from concurrent.futures import ThreadPoolExecutor


def plot_results(runs_metrics, window_size=500, interpolation_resolution=100_000,
                 nametag="",
                 fig=None, ax=None, color="blue",
                 avg_label=None, individual_label="", plot_individual=True):
    """
    Plots the episodic returns for multiple runs, including individual run curves and the average curve.
    
    Parameters:
        runs_metrics (list): A list of runs, each being a list of dictionaries containing
                             "episode_return" and "episode_length".
        window_size (int): Window size for the moving average smoothing.
        interpolation_resolution (int): Number of points for interpolation.
        nametag (str): Tag to include in the saved filename.
        fig (matplotlib.figure.Figure, optional): Figure object to plot on.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on.
        avg_label (str, optional): Label for the average return curve.
        individual_label (str, optional): Label for the individual runs.
    
    Returns:
        fig, ax: The matplotlib Figure and Axes objects.
    """
    
    # Create a new figure and axis if not provided.
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.tight_layout(pad=3.0)
    else:
        plt.tight_layout()

    # Extract episode returns and lengths.
    episode_return_lst = [[ep.get("episode_return") for ep in run] for run in runs_metrics]
    episode_length_lst = [[ep.get("episode_length") for ep in run] for run in runs_metrics]
    
    # Determine the minimum number of steps among runs.
    num_steps = min([sum(lengths) for lengths in episode_length_lst])
    x = np.linspace(0, num_steps, interpolation_resolution)
    returns_lst = []

    # Plot each run's reward curve.
    for i, (ep_returns_list, ep_lengths_list) in enumerate(zip(episode_return_lst, episode_length_lst)):
        ep_returns = np.array(ep_returns_list, dtype=float)
        ep_lengths = np.array(ep_lengths_list, dtype=float)
        ep_timesteps = np.cumsum(ep_lengths)
        
        # Interpolate the episodic returns over the x-axis.
        ep_returns_interpolated = np.interp(x, ep_timesteps, ep_returns)
        
        # Apply a moving average for smoothing.
        ep_returns_interpolated_smooth = np.empty_like(ep_returns_interpolated)
        for j in range(len(ep_returns_interpolated)):
            start_idx = max(0, j - window_size)
            end_idx = min(len(ep_returns_interpolated), j + window_size)
            ep_returns_interpolated_smooth[j] = np.mean(ep_returns_interpolated[start_idx:end_idx])
        
        # Plot the individual run curve.
        if plot_individual:
            if i == 0:
                ilabel = individual_label if individual_label!="" else "Individual Runs"
                ax.plot(x, ep_returns_interpolated_smooth, '-', alpha=0.2, markersize=1, label=ilabel, color=color)
            else:
                ax.plot(x, ep_returns_interpolated_smooth, '-', alpha=0.2, markersize=1, color=color)
        returns_lst.append(ep_returns_interpolated_smooth)
    
    returns_lst = np.asarray(returns_lst)
    avg_returns = np.mean(returns_lst, axis=0)
    
    # Option A
    # lower = np.percentile(returns_lst, 2.5, axis=0)
    # upper = np.percentile(returns_lst, 97.5, axis=0)
    # ax.fill_between(x,
    #                 lower,
    #                 upper,
    #                 color=color,
    #                 alpha=0.2,
    #                 )

    # Option B
    n = returns_lst.shape[0]
    sem = np.std(returns_lst, axis=0, ddof=1) / np.sqrt(n)      # standard error
    # ci_margin = 1.96 * sem                                     # 95% CI ≈ mean ± 1.96·SEM
    # ci_margin = 1.645 * sem                                     # 90% CI ≈ mean ± 1.645·SEM
    ci_margin = 1.036 * sem                                     # 70% CI ≈ mean ± 1.036·SEM

    ax.fill_between(x,
                    avg_returns - ci_margin,
                    avg_returns + ci_margin,
                    color=color,
                    alpha=0.2,
                    )

    
    # Define the label for the average return curve.
    if avg_label is None:
        avg_label = f"Avg Return Over {len(returns_lst)} Runs"
    
    # Plot the average return.
    ax.plot(x, avg_returns, linestyle="-", label=avg_label, color=color)
    
    # Set plot properties.
    ax.set_title(f"Result of {nametag}")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.grid(True)
    ax.legend()
    if nametag is not None:
        fig.savefig(f"{nametag}.png")
    
    return fig, ax

def generate_video(folders, episode, fps=30):
    """
    Given the loaded metrics (list of runs → list of episode dicts),
    write out a GIF of the frames for `run_index`-th run and `episode`-th episode.

    Args:
        run_metrics (List[List[dict]]): output of load_results(), i.e.
            run_metrics[run][episode]["frames"] is a List[np.ndarray].
        episode (int): which episode (index) to visualize.
        run_index (int): which run (index) to pick. Default 0.
        out_dir (str): directory to write the .gif into.
        fps (int): frames per second for the GIF.

    Returns:
        str: path to the written GIF file.
    """
    
    for folder in folders:
        with open(f"{folder}/res.pkl", "rb") as f:
            run = pickle.load(f)

        frames = run[episode].get("frames", None)
        if frames is None:
            raise KeyError(f"No 'frames' key in run_metrics[{episode}]")

        # ensure output dir exists
        gif_path = os.path.join(folder, f"ep{episode}.gif")

        # write out
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved episode {episode} to {gif_path}")


def load_results(args):
    # Create a file pattern based on environment name and max_steps.
    pattern = f"{args.res_dir}/{args.pattern}"
    folders = glob.glob(pattern)
    if not folders:
        print(f"No experiment found for pattern: {pattern}")
        return False
    
    runs_metrics = []
    for dir in folders:
        with open(f"{dir}/res.pkl", "rb") as f:
            run_result = pickle.load(f)
            runs_metrics.append(run_result)
        print(f"Loaded {dir}/res.pkl")
    return runs_metrics, folders


def plot_comparison(method_patterns, 
                    res_dir,
                    window_size,
                    interpolation_resolution,
                    out_fname="method_comparison.png"):
    """
    Overlay the average return curves of multiple methods on a single plot.
    """
    # create shared figure/axes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.tight_layout(pad=3.0)

    # automatic distinct colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (method_name, pattern), color in zip(method_patterns.items(), colors):
        print(f"Loading {method_name}")
        folders = glob.glob(f"{res_dir}/{pattern}")
        print(f"{len(folders)} experiments found")
        if not folders:
            print(f"Warning: no folders match pattern {res_dir}/{pattern}")
            continue

        # load each run's res.pkl
        runs = []
        for folder in folders:
            try:
                with open(os.path.join(folder, "res.pkl"), "rb") as f:
                    r = pickle.load(f)
                    runs.append(r)
            except:
                print(f"Unable to open the file: {os.path.join(folder, 'res.pkl')}")

        if len(runs) == 0:
            continue
        # overlay the average return curve
        plot_results(
            runs,
            window_size=window_size,
            interpolation_resolution=interpolation_resolution,
            nametag=None,         # skip per‐method saving
            fig=fig, ax=ax,
            color=color,
            avg_label=method_name,
            individual_label=None,
            plot_individual=False
        )

    # finalize styling
    ax.set_title(out_fname)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.grid(True)
    ax.legend(loc="best")

    # save once:li
    fig.savefig(out_fname, format="svg")
    return fig, ax