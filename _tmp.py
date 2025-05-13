import os
import re
import pandas as pd
from glob import glob
from collections import defaultdict

# Regex patterns
job_pattern = re.compile(r"Running job \d+ \(exp=(\d+), seed=(\d+)\)")
lr_pattern = re.compile(r"Learning rate: ([\d.]+)")
clip_pattern = re.compile(r"Clip coef: ([\d.]+)")
ent_pattern = re.compile(r"Entropy coef: ([\d.]+)")
step_pattern = re.compile(r"Optimal trajectory found on step (\d+)")

# Directory containing .out files
directory = "./"  # change if needed

# Dict to hold: (exp, lr, clip, ent) -> {seed: (step, filename)}
grouped_results = defaultdict(lambda: {})

for filepath in glob(os.path.join(directory, "*.out")):
    with open(filepath, "r") as file:
        lines = file.readlines()

    exp, seed = None, None
    lr = clip = ent = None
    step = float("inf")

    for line in lines:
        if (match := job_pattern.search(line)):
            exp, seed = int(match.group(1)), int(match.group(2))
        elif (match := lr_pattern.search(line)):
            lr = float(match.group(1))
        elif (match := clip_pattern.search(line)):
            clip = float(match.group(1))
        elif (match := ent_pattern.search(line)):
            ent = float(match.group(1))
        elif (match := step_pattern.search(line)):
            step = int(match.group(1))

    if None not in (exp, seed, lr, clip, ent):
        key = (exp, lr, clip, ent)
        grouped_results[key][seed] = (step, os.path.basename(filepath))

# Build final dataframe
records = []
for (exp, lr, clip, ent), seed_data in grouped_results.items():
    steps = []
    files = []
    for seed in [0, 1, 2]:
        step, fname = seed_data.get(seed, (float("inf"), ""))
        steps.append(step)
        files.append(fname)

    avg_step = sum(steps) / len(steps)
    records.append({
        "exp": exp,
        "learning_rate": lr,
        "clip_coef": clip,
        "entropy_coef": ent,
        "seed_0_step": steps[0],
        "seed_0_file": files[0],
        "seed_1_step": steps[1],
        "seed_1_file": files[1],
        "seed_2_step": steps[2],
        "seed_2_file": files[2],
        "avg_step": avg_step
    })

df = pd.DataFrame(records)
df = df.sort_values(by="avg_step", ascending=True).reset_index(drop=True)

# Print and save
print(df)
df.to_csv("_averaged_results_with_files.csv", index=False)