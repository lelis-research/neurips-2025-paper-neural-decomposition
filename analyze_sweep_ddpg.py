#!/usr/bin/env python
import os
import re
import pickle
import numpy as np
import pandas as pd

# — adjust this to your DDPG sweep output directory:
ROOT_DIR = "Results_car_ddpg/"

# Regex to extract your hyper‐params from folder names:
PAT = re.compile(
    r'^(?P<env>.+?)_'        # e.g. "car-train"
    r'(?P<seed>\d+)_'        # e.g. "1000"
    r'(?P<steps>\d+)_'       # e.g. "2000000"
    r'al_(?P<actor_lr>[\d\.eE+-]+)_'
    r'cl_(?P<critic_lr>[\d\.eE+-]+)_'
    r'buf_(?P<buf_size>\d+)_'
    r'bs_(?P<batch_size>\d+)_'
    r'th_(?P<ou_theta>[\d\.eE+-]+)_'
    r'sig_(?P<ou_sigma>[\d\.eE+-]+)$'
)

records = []
for name in os.listdir(ROOT_DIR):
    m = PAT.match(name)
    if not m:
        continue
    p = m.groupdict()
    combo = {
        "actor_lr":   float(p["actor_lr"]),
        "critic_lr":  float(p["critic_lr"]),
        "buf_size":   int(p["buf_size"]),
        "batch_size": int(p["batch_size"]),
        "ou_theta":   float(p["ou_theta"]),
        "ou_sigma":   float(p["ou_sigma"]),
    }
    seed = int(p["seed"])
    res_path = os.path.join(ROOT_DIR, name, "res.pkl")
    if not os.path.isfile(res_path):
        print(f"⚠️  Missing res.pkl in {name}, skipping")
        continue
    try:
        with open(res_path, "rb") as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"⚠️  Couldn’t load {res_path}: {e}")
        continue

    # compute AUC of episode returns
    returns = [ep["episode_return"] for ep in results]
    auc = np.trapz(returns, dx=1) / len(returns)

    rec = {**combo, "seed": seed, "auc": auc, "file_name": name}
    records.append(rec)

# long DataFrame
df_long = pd.DataFrame(records)

# aggregate file_name per hyper‐param combo
file_names = (
    df_long
      .groupby(["actor_lr","critic_lr","buf_size","batch_size","ou_theta","ou_sigma"])["file_name"]
      .apply(lambda x: ";".join(sorted(x.unique())))
      .reset_index()
)

# pivot to wide: one column per seed’s AUC
df_wide = df_long.pivot_table(
    index=["actor_lr","critic_lr","buf_size","batch_size","ou_theta","ou_sigma"],
    columns="seed",
    values="auc"
)
df_wide.columns = [f"auc_seed_{col}" for col in df_wide.columns]

# summary stats across seeds
df_wide["mean_auc"] = df_wide.mean(axis=1)
df_wide["std_auc"]  = df_wide.std(axis=1, ddof=1)
df_wide["min_auc"]  = df_wide.min(axis=1)
df_wide["max_auc"]  = df_wide.max(axis=1)
df_wide["n_seeds"]  = df_wide.count(axis=1)

# reset index and sort by best mean
df_sorted = df_wide.reset_index().sort_values("mean_auc", ascending=False)

# merge in the aggregated file names
df_final = df_sorted.merge(
    file_names,
    on=["actor_lr","critic_lr","buf_size","batch_size","ou_theta","ou_sigma"],
    how="left"
)

# show & save
pd.set_option("display.precision", 6)
print(df_final)
df_final.to_csv("sweep_ddpg_summary.csv", index=False)
print("→ Saved summary to sweep_ddpg_summary.csv")