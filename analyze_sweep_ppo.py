#!/usr/bin/env python
import os
import re
import pickle
import numpy as np
import pandas as pd

# Adjust this to wherever your sweep folders live:
ROOT_DIR = "Results_car_top_action_l1reg/"

# Regex to extract your hyper‐params
PAT = re.compile(
    r'^(?P<env>.+?)_'
    r'(?P<seed>\d+)_'
    r'(?P<steps>\d+)_'
    r'ss_(?P<step_size>[\d\.eE+-]+)_'
    r'm_(?P<num_minibatches>\d+)_'
    r'r_(?P<rollout_steps>\d+)_'
    r'e_(?P<entropy_coef>[\d\.eE+-]+)_'
    r'l1_(?P<l1_lambda>[\d\.eE+-]+)$'
)

records = []
for name in os.listdir(ROOT_DIR):
    m = PAT.match(name)
    if not m:
        continue
    params = m.groupdict()
    combo = {
        "step_size":       float(params['step_size']),
        "num_minibatches": int(params['num_minibatches']),
        "rollout_steps":   int(params['rollout_steps']),
        "entropy_coef":    float(params['entropy_coef']),
        "l1_lambda":    float(params['l1_lambda']),

    }
    seed = int(params['seed'])
    res_path = os.path.join(ROOT_DIR, name, "res.pkl")
    if not os.path.isfile(res_path):
        print(f"⚠️  Missing res.pkl in {name}, skipping")
        continue
    try:
        with open(res_path, "rb") as f:
            results = pickle.load(f)
    except:
        print("couldn't load")
        continue
    returns = [ep["episode_return"] for ep in results]
    auc = np.trapz(returns, dx=1) / len(returns)

    # include the folder name
    rec = {
        **combo,
        "seed":      seed,
        "auc":       auc,
        "file_name": name
    }
    records.append(rec)

# long DataFrame
df_long = pd.DataFrame(records)

# aggregate file_name per hyper‐param combo
file_names = (
    df_long
    .groupby(["step_size","num_minibatches","rollout_steps","entropy_coef","l1_lambda"])["file_name"]
    .apply(lambda x: ";".join(sorted(x.unique())))
    .reset_index()
)

# pivot to wide
df_wide = df_long.pivot_table(
    index=["step_size","num_minibatches","rollout_steps","entropy_coef", "l1_lambda"],
    columns="seed",
    values="auc"
)
df_wide.columns = [f"auc_seed_{col}" for col in df_wide.columns]

# summary stats
df_wide["mean_auc"] = df_wide.mean(axis=1)
df_wide["std_auc"]  = df_wide.std(axis=1, ddof=1)
df_wide["min_auc"]  = df_wide.min(axis=1)
df_wide["max_auc"]  = df_wide.max(axis=1)
df_wide["n_seeds"]  = df_wide.count(axis=1)

# reset index and sort
df_sorted = df_wide.reset_index().sort_values("mean_auc", ascending=False)

# merge in the aggregated file names
df_final = df_sorted.merge(
    file_names,
    on=["step_size","num_minibatches","rollout_steps","entropy_coef"],
    how="left"
)

# save
pd.set_option("display.precision", 6)
print(df_final)
df_final.to_csv("sweep_car_top_action_l1.csv", index=False)
# print("\nSaved summary (wide) with file_name column to sweep_summary_with_filenames.csv")