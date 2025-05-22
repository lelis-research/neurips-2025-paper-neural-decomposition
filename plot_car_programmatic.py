# plot_from_excel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your data. 
#    If your Excel has no seed column, we’ll just use the row-number as a label.
df = pd.read_csv("car_nstepdqn.csv")  # assumes columns “train” and “test”

# 2) (Optional) drop rows with zero train success
# df = df[df["train"] > 0]

# 3) Sort by descending train success
df = df.sort_values("train", ascending=False)

# 4) Take top 30 (if you have >30 rows)
df = df.head(30)

# 5) Prepare for plotting
#    Use the existing index as labels, or if you have a seed column, replace these
labels = df.index.astype(str).tolist()
train_scores = df["train"].values
test_scores  = df["test"].values

x = np.arange(len(labels))
width = 0.4

plt.figure(figsize=(14,6))
plt.bar(x - width/2, train_scores, width, label="car-train")
plt.bar(x + width/2, test_scores,  width, label="car-test")
plt.xticks(x, labels, rotation=90)
plt.xlabel("Row index (top 30 by train success)")
plt.ylabel("Successful parks out of 100")
plt.title("Top 30 Results from Excel")
plt.legend()
plt.tight_layout()
plt.savefig("success_top30_nstepdqn.png")
print("Saved plot to success_top30_prog.png")