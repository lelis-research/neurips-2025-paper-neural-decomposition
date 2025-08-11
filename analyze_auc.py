import os
import torch
import re
import numpy as np
from collections import defaultdict
from pipelines.train_ppo import Args

# Root path to your model folders
base_dir = "binary"

# Settings and environment seeds
settings = ["internal", "both", "input"]
env_seeds = [51,8]

# Dictionary to hold AUCs
auc_results = defaultdict(lambda: defaultdict(list))  # {(setting, env_seed): {hyperparam_string: [auc1, auc2, ...]}}

# Regex to extract hyperparameters from filename
pattern = re.compile(
    r"train_ppoAgent_MiniGrid-FourRooms-v0_gw9_h64_lr(?P<lr>[0-9.]+)_clip(?P<clip>[0-9.]+)_ent(?P<ent>[0-9.]+)_envsd(?P<envsd>\d+).pt"
)

# Walk through each setting and env_seed
for setting in settings:
    for env_seed in env_seeds:
        folder_glob = f"models_sweep_MiniGrid-FourRooms-v0_{env_seed}_didec_{setting}_"
        for folder_name in os.listdir(base_dir):
            if folder_name.startswith(folder_glob):
                full_folder_path = os.path.join(base_dir, folder_name)
                reg_match = re.search(r"_([0-9.]+)$", folder_name)
                if not reg_match:
                    continue
                reg_coef = float(reg_match.group(1))

                # Go into each seed subfolder
                for seed_folder in os.listdir(full_folder_path):
                    print(seed_folder)
                    exit()
                    seed_path = os.path.join(full_folder_path, seed_folder)
                    if not os.path.isdir(seed_path):
                        continue

                    # Load each model file
                    for filename in os.listdir(seed_path):
                        if filename.endswith(".pt") and filename.startswith("train_ppoAgent"):
                            match = pattern.match(filename)
                            if not match:
                                continue
                            lr = float(match.group("lr"))
                            clip = float(match.group("clip"))
                            ent = float(match.group("ent"))

                            model_path = os.path.join(seed_path, filename)
                            try:
                                data = torch.load(model_path, map_location="cpu", weights_only=False)
                                steps = np.array(data["steps"])
                                returns = np.array(data["average_returns"])
                                if len(steps) != len(returns) or len(steps) < 2:
                                    continue
                                auc = np.trapz(returns, steps)

                                hyperparam_key = (reg_coef, lr, clip, ent)
                                auc_results[(setting, env_seed)][hyperparam_key].append(auc)
                            except Exception as e:
                                print(f"Failed loading {model_path}: {e}")

# Compute average AUC and find best config per (setting, env_seed)
best_config_per_combo = {}

for key, auc_dict in auc_results.items():
    setting, env_seed = key
    best_auc = -np.inf
    best_hyperparam = None
    for hyperparam, auc_list in auc_dict.items():
        avg_auc = np.mean(auc_list)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_hyperparam = hyperparam
    best_config_per_combo[(setting, env_seed)] = {
        "reg_coef": best_hyperparam[0],
        "learning_rate": best_hyperparam[1],
        "clipping_coef": best_hyperparam[2],
        "entropy_coef": best_hyperparam[3],
        "average_auc": best_auc
    }

# Display results
# Display results using a simple for loop
print("\nBest Hyperparameter Configurations per (Setting, Env Seed):\n")
for (setting, env_seed), config in best_config_per_combo.items():
    print(f"Setting: {setting}, Env Seed: {env_seed}")
    print(f"  - reg_coef      : {config['reg_coef']}")
    print(f"  - learning_rate : {config['learning_rate']}")
    print(f"  - clipping_coef : {config['clipping_coef']}")
    print(f"  - entropy_coef  : {config['entropy_coef']}")
    print(f"  - average AUC   : {config['average_auc']:.2f}")
    print("-" * 50)


