import os
import re
import pandas as pd
from collections import defaultdict

# Patterns to extract settings and log info
settings_pattern = re.compile(
    r"Running job.*\(exp=\d+, seed=(\d+)\).*?\n\s*Learning rate: ([\d.]+)\n\s*Clip coef: ([\d.]+)\n\s*Entropy coef: ([\d.]+)",
    re.MULTILINE
)

log_pattern = re.compile(
    r"global_step=(\d+), episodic_return=([\d\-.]+)"
)

# Store results
results = []

for filename in os.listdir():
    if filename.endswith(".out"):
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract settings
        match = settings_pattern.search(content)
        if not match:
            print(f"Skipping file (settings not found): {filename}")
            continue
        
        seed, lr, clip, ent = match.groups()
        seed, lr, clip, ent = int(seed), float(lr), float(clip), float(ent)
        
        # Extract reward logs
        steps_rewards = [(0, 0.0)]
        for step_match in log_pattern.finditer(content):
            step = int(step_match.group(1))
            reward = float(step_match.group(2))
            steps_rewards.append((step, reward))
        
        # Compute AUC
        auc = 0.0
        for i in range(1, len(steps_rewards)):
            prev_step, _ = steps_rewards[i - 1]
            curr_step, reward = steps_rewards[i]
            auc += reward * (curr_step - prev_step)
        
        results.append({
            'file': filename,
            'seed': seed,
            'learning_rate': lr,
            'clip_coef': clip,
            'entropy_coef': ent,
            'auc': auc
        })

# Create raw results DataFrame
df = pd.DataFrame(results)

# Group by hyperparams and aggregate
grouped = df.groupby(['learning_rate', 'clip_coef', 'entropy_coef']).agg(
    avg_auc=('auc', 'mean'),
    runs=('auc', 'count'),
    files=('file', lambda x: ', '.join(sorted(x)))
).reset_index()

# Sort by average AUC
grouped = grouped.sort_values(by='avg_auc', ascending=False)

# Save to CSV
grouped.to_csv("_grouped_auc_results.csv", index=False)

print(grouped)
# Display in notebook or UI
# import ace_tools as tools; tools.display_dataframe_to_user(name="Grouped AUC Results with File Names", dataframe=grouped)