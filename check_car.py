#!/usr/bin/env python
import os
import glob
import re
import pickle
import numpy as np
import pandas as pd

# adjust this to the directory containing your car-train_*_... folders
ROOT_DIR = "Results/"

# glob for anything matching car-train_<seed>_4000000_all_actions
pattern = os.path.join(ROOT_DIR, "car-train_*_4000000_all_actions")
dirs = glob.glob(pattern)

# regex to pull the seed out of the folder name
seed_re = re.compile(r"car-train_(\d+)_4000000_all_actions$")

records = []
for d in dirs:
    name = os.path.basename(d)
    m = seed_re.match(name)
    if not m:
        continue
    seed = int(m.group(1))

    res_path = os.path.join(d, "res.pkl")
    if not os.path.isfile(res_path):
        print(f"⚠️  Missing res.pkl in {name}, skipping")
        continue

    # load the list of {"episode_return": …}
    with open(res_path, "rb") as f:
        results = pickle.load(f)

    # extract returns and compute mean
    returns = [ ep["episode_return"] for ep in results ]
    mean_return = float(np.mean(returns))

    records.append({
        "seed": seed,
        "mean_return": mean_return,
        "path": d
    })

# build and sort DataFrame
df = pd.DataFrame(records)
df_sorted = df.sort_values("mean_return", ascending=False).reset_index(drop=True)

# display
pd.set_option("display.precision", 6)
print(df_sorted)

# optionally save
df_sorted.to_csv("car_train_mean_returns.csv", index=False)
print("\nSaved summary to car_train_mean_returns.csv")