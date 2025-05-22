# aggregate_and_plot_top30.py

import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Read per-seed results
data = {}
for fn in glob.glob("Results_car_dqn_best_csv/results_*.csv"):
    seed = int(os.path.basename(fn).split("_")[1].split(".")[0])
    with open(fn) as f:
        train_s, test_s = f.read().strip().split(",")
    data[seed] = (int(train_s), int(test_s))

# 2) Build DataFrame
df = pd.DataFrame(data, index=["car-train", "car-test"])

# 3) Filter out seeds with zero train success
# df = df.loc[:, df.loc["car-train"] > 0]

# 4) Pick top 30 seeds by car-train success
top_seeds = (
    df
    .loc["car-train"]
    .sort_values(ascending=False)   # highest first
    .head(30)                       # take top 30
    .index
)
df_top30 = df[top_seeds]

# 5) (Optional) Save filtered CSV
df_top30.to_csv("top30_results.csv")

# 6) Plot
seeds        = list(df_top30.columns)
train_scores = df_top30.loc["car-train"].values
test_scores  = df_top30.loc["car-test"].values

x = np.arange(len(seeds))
width = 0.4

plt.figure(figsize=(14,6))
plt.bar(x - width/2, train_scores, width, label="car-train")
plt.bar(x + width/2, test_scores,  width, label="car-test")
plt.xticks(x, seeds, rotation=90)
plt.xlabel("Random seed (top 30 by car-train success)")
plt.ylabel("Successful parks out of 100")
plt.title("Top 30 Seeds by car-train Performance")
plt.legend()
plt.tight_layout()
plt.savefig("success_top30_dqn.png")

print("Wrote top30_results.csv and success_top30_dqn.png")