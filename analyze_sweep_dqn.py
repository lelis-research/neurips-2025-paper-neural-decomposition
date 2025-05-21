#!/usr/bin/env python
import os
import re
import pickle
import numpy as np
import pandas as pd

# Directory where all sweep results are stored
ROOT_DIR = "Results_car_dqn/"

# Regex pattern to extract hyperparameters
PAT = re.compile(
    r'^car-train_'
    r'(?P<seed>\d+)_'
    r'(?P<steps>\d+)_'
    r'ss_(?P<ss>[\d\.eE+-]+)_'
    r'bs_(?P<bs>\d+)_'
    r'tu_(?P<tu>\d+)_'
    r'e_(?P<e>[\d\.eE+-]+)_'
    r'rb_(?P<rb>\d+)_'
    r'ar_(?P<ar>\d+)$'
)

records = []
for name in os.listdir(ROOT_DIR):
    m = PAT.match(name)
    if not m:
        continue
    params = m.groupdict()
    combo = {
        "ss": float(params['ss']),
        "bs": int(params['bs']),
        "tu": int(params['tu']),
        "e":  float(params['e']),
        "rb": int(params['rb']),
        "ar": int(params['ar']),
    }
    seed = int(params['seed'])

    res_path = os.path.join(ROOT_DIR, name, "res.pkl")
    if not os.path.isfile(res_path):
        print(f"⚠️  Missing res.pkl in {name}, skipping")
        continue
    try:
        with open(res_path, "rb") as f:
            results = pickle.load(f)
    except Exception as ex:
        print(f"❌ Couldn't load {res_path}: {ex}")
        continue

    returns = [ep["episode_return"] for ep in results]
    auc = np.trapz(returns, dx=1) / len(returns)

    rec = {
        **combo,
        "seed": seed,
        "auc": auc,
        "file_name": name
    }
    records.append(rec)

# Create long format DataFrame
df_long = pd.DataFrame(records)

# Aggregate file names by hyperparameter combination
file_names = (
    df_long
    .groupby(["ss", "bs", "tu", "e", "rb", "ar"])["file_name"]
    .apply(lambda x: ";".join(sorted(x.unique())))
    .reset_index()
)

# Convert to wide format with separate columns per seed
df_wide = df_long.pivot_table(
    index=["ss", "bs", "tu", "e", "rb", "ar"],
    columns="seed",
    values="auc"
)
df_wide.columns = [f"auc_seed_{col}" for col in df_wide.columns]

# Summary statistics
df_wide["mean_auc"] = df_wide.mean(axis=1)
df_wide["std_auc"]  = df_wide.std(axis=1, ddof=1)
df_wide["min_auc"]  = df_wide.min(axis=1)
df_wide["max_auc"]  = df_wide.max(axis=1)
df_wide["n_seeds"]  = df_wide.count(axis=1)

# Merge and sort
df_sorted = df_wide.reset_index().sort_values("mean_auc", ascending=False)
df_final = df_sorted.merge(file_names, on=["ss", "bs", "tu", "e", "rb", "ar"], how="left")

# Print and save
pd.set_option("display.precision", 6)
print(df_final)
df_final.to_csv("sweep_car_car.csv", index=False)