import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle



def plot_results(runs_metrics, window_size=500, interpolation_resolution=100_000,
                 nametag="",
                 fig=None, ax=None, color="blue",
                 avg_label=None, individual_label=""):
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
        if i == 0:
            ilabel = individual_label if individual_label!="" else "Individual Runs"
            ax.plot(x, ep_returns_interpolated_smooth, '-', alpha=0.2, markersize=1, label=ilabel, color=color)
        else:
            ax.plot(x, ep_returns_interpolated_smooth, '-', alpha=0.2, markersize=1, color=color)
        returns_lst.append(ep_returns_interpolated_smooth)
    
    returns_lst = np.asarray(returns_lst)
    avg_returns = np.mean(returns_lst, axis=0)
    
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
    
    fig.savefig(f"{nametag}.png")
    
    return fig, ax


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
    return runs_metrics
    

