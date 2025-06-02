import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
ENTITY = "rabdolla-uuniversity-of-alberta"
PROJECT = "NEURIPS_2025_test2"
METRIC = "Charts/episodic_return_avg"
STEP = "_step"
SMOOTH_WINDOW = 100
DOWNSAMPLE_STEP = 10  # plot every 100 steps
NUM_POINTS = 1000  # Number of step samples to interpolate to (uniform grid)

# --- Connect to API ---
wandb.login(key="af92d00a13698da89f8ff2ae5e2d8bc4d932e26a")
api = wandb.Api()

# --- Fetch runs ---
runs = api.runs(f"{ENTITY}/{PROJECT}")

# --- Organize runs by group ---
grouped = defaultdict(list)
for run in runs:
    if run.state != "finished":
        continue
    group = run.group or run.name
    grouped[group].append(run)

# --- Prepare and plot ---
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

for group_name, run_list in grouped.items():
    # if not ("input" in group_name):
    #     continue
    dfs = []
    min_step = float('inf')
    max_step = float('-inf')

    # First pass: collect and smooth each run
    for run in run_list:
        history = run.history(keys=[STEP, METRIC], pandas=True)
        history = history.dropna(subset=[STEP, METRIC])
        if not history.empty:
            smoothed = history[[STEP, METRIC]].copy()
            smoothed[METRIC] = smoothed[METRIC].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            min_step = min(min_step, smoothed[STEP].min())
            max_step = max(max_step, smoothed[STEP].max())
            dfs.append(smoothed)

    # print(dfs)
    if not dfs:
        continue

    # Create common step grid
    common_steps = np.linspace(min_step, max_step, NUM_POINTS)

    # Interpolate each smoothed run onto common steps
    interp_runs = []
    for df in dfs:
        interp = np.interp(common_steps, df[STEP], df[METRIC])
        interp_runs.append(interp)

    # Convert to DataFrame for easier stat computation
    interp_df = pd.DataFrame(interp_runs, columns=common_steps)
    # print (interp_df)

    # Compute mean and 95% CI
    mean = interp_df.mean(axis=0)
    sem = interp_df.sem(axis=0)
    lower = mean - 1.96 * sem
    upper = mean + 1.96 * sem

    # print("Mean", mean)
    # print("Lower CI", lower)
    # print("Upper CI", upper)
    # print("Common steps", common_steps)
    # print(sem)

    # print(max_step, min_step)
    # print(DOWNSAMPLE_STEP * NUM_POINTS)
    # print((DOWNSAMPLE_STEP * NUM_POINTS // (max_step - min_step)))
    # print(np.arange(len(common_steps)))

    # Downsample for plotting
    # mask = np.arange(len(common_steps)) % (DOWNSAMPLE_STEP * NUM_POINTS // (max_step - min_step)) == 0
    mask = np.arange(len(common_steps)) > -1
    plot_steps = common_steps[mask]
    plot_mean = mean[mask]
    plot_lower = lower[mask]
    plot_upper = upper[mask]

    # print("Plot steps", plot_steps)
    # print("Plot mean", plot_mean)
    # print("Plot lower", plot_lower)
    # print("Plot upper", plot_upper)
    # print("Plot mask", mask)
    # print("Plot mask length", len(mask))
    # print("Plot mean length", len(plot_mean))
    # print("Plot lower length", len(plot_lower))
    # print("Plot upper length", len(plot_upper))
    # print("Plot steps length", len(plot_steps))
    # Ensure lengths match

    # Plot
    plt.plot(plot_steps, plot_mean, label=group_name)
    plt.fill_between(plot_steps, plot_lower, plot_upper, alpha=0.3)



plt.xlabel("Step")
plt.ylabel("Smoothed Episodic Return")
plt.title("Interpolated & Smoothed Episodic Return with 95% CI")
plt.legend()
plt.tight_layout()
plt.show()