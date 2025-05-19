#!/usr/bin/env python
import os
import re
import pickle
import numpy as np
import pandas as pd

# Adjust this to wherever your sweep folders live:
ROOT_DIR = "Results/"

# Regex to extract (env, seed, total_steps, step_size, num_minibatches, rollout_steps, entropy_coef)
PAT = re.compile(
    r'^(?P<env>.+?)_'             # env name (e.g. car-train)
    r'(?P<seed>\d+)_'             # seed
    r'(?P<steps>\d+)_'            # total steps
    r'ss_(?P<step_size>[\d\.eE+-]+)_'
    r'm_(?P<num_minibatches>\d+)_'
    r'r_(?P<rollout_steps>\d+)_'
    r'e_(?P<entropy_coef>[\d\.eE+-]+)$'
)

# Collect one record per (config, seed)
records = []

for name in os.listdir(ROOT_DIR):
    m = PAT.match(name)
    if not m:
        continue
    params = m.groupdict()
    combo = {
        "step_size":        float(params['step_size']),
        "num_minibatches":  int(params['num_minibatches']),
        "rollout_steps":    int(params['rollout_steps']),
        "entropy_coef":     float(params['entropy_coef']),
    }
    seed = int(params['seed'])
    res_path = os.path.join(ROOT_DIR, name, "res.pkl")
    if not os.path.isfile(res_path):
        print(f"⚠️  Missing res.pkl in {name}, skipping")
        continue

    # Load returns and compute AUC
    with open(res_path, "rb") as f:
        results = pickle.load(f)
    returns = [ ep["episode_return"] for ep in results ]
    
    # auc = np.trapz(returns, dx=1)
    
    num_episodes = len(returns)
    auc = np.trapz(returns, dx=1) / num_episodes

    # one record per seed
    rec = {
        **combo,
        "seed": seed,
        "auc": auc
    }
    records.append(rec)

# Turn into DataFrame
df_long = pd.DataFrame(records)

# Pivot to wide: one row per hyper‐param combo, one column per seed
df_wide = df_long.pivot_table(
    index=["step_size","num_minibatches","rollout_steps","entropy_coef"],
    columns="seed",
    values="auc"
)

# Rename columns to something like auc_seed_10000, etc.
df_wide.columns = [f"auc_seed_{col}" for col in df_wide.columns]

# Compute summary stats across those seed‐columns
df_wide["mean_auc"] = df_wide.mean(axis=1)
df_wide["std_auc"]  = df_wide.std(axis=1, ddof=1)
df_wide["min_auc"]  = df_wide.min(axis=1)
df_wide["max_auc"]  = df_wide.max(axis=1)
df_wide["n_seeds"]  = df_wide.count(axis=1)

# Sort by mean_auc descending
df_sorted = df_wide.sort_values("mean_auc", ascending=False)

# Reset index so hyper-params become columns again
df_sorted = df_sorted.reset_index()

# Display / save
pd.set_option("display.precision", 6)
print(df_sorted)

df_sorted.to_csv("sweep_summary_wide.csv", index=False)
print("\nSaved summary (wide) to sweep_summary_wide.csv")