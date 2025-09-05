import os
import torch
import re
import json
import numpy as np
from collections import defaultdict
from pipelines.train_ppo import Args

# Root path to your model folders
base_dir = "binary"
def analyze_auc(settings=["both"], env_seeds=[12], option_mode="vanilla", config_filepath="binary/configs/ComboGrid_gw5.json"):
    # Dictionary to hold AUCs
    auc_results = defaultdict(lambda: defaultdict(list))  # {(setting, env_seed): {hyperparam_string: [auc1, auc2, ...]}}

    # Regex to extract hyperparameters from filename
    pattern = re.compile(
        r"train_ppoAgent_ComboGrid_gw6_h64_lr(?P<lr>[0-9.]+)_clip(?P<clip>[0-9.]+)_ent(?P<ent>[0-9.]+)_envsd(?P<envsd>\d+)_BL-MR-ML-BM-TM-combo4.pt"
    )

    # Walk through each setting and env_seed
    for setting in settings:
        for env_seed in env_seeds:
            folder_glob = f"models_sweep_ComboGrid_{env_seed}_{option_mode}{'_'+setting+'_' if option_mode == 'didec' else ''}"
            for folder_name in os.listdir(base_dir):
                if folder_name.startswith(folder_glob):
                    full_folder_path = os.path.join(base_dir, folder_name)
                    reg_match = re.search(r"_([0-9.]+)$", folder_name)
                    if not reg_match:
                        reg_coef = 0
                    else:
                        reg_coef = float(reg_match.group(1))

                    # Go into each seed subfolder
                    for seed_folder in os.listdir(full_folder_path):
                        seed_path = os.path.join(full_folder_path, seed_folder)
                        if not os.path.isdir(seed_path):
                            continue
                        # Load each model file
                        for filename in os.listdir(seed_path):
                            if filename.endswith("combo4.pt") and filename.startswith("train_ppoAgent"):
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
                                    auc_results[(option_mode, setting)][hyperparam_key].append(auc)
                                except Exception as e:
                                    print(f"Failed loading {model_path}: {e}")

    # Compute average AUC and find best config per (setting, env_seed)
    best_config_per_combo = {}

    for key, auc_dict in auc_results.items():
        option_mode, setting = key
        best_auc = -np.inf
        best_hyperparam = None
        for hyperparam, auc_list in auc_dict.items():
            avg_auc = np.mean(auc_list)
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_hyperparam = hyperparam
        
        best_config_per_combo[f"{option_mode}_{setting}"] = {
            "reg_coef": best_hyperparam[0],
            "learning_rate": best_hyperparam[1],
            "clipping_coef": best_hyperparam[2],
            "entropy_coef": best_hyperparam[3],
            "average_auc": best_auc
        }

    data = {}

    try:
        os.makedirs("binary/configs", exist_ok=True)
    except OSError as error:
        pass
    
    try:
        with open(config_filepath, 'r') as f:
            data = json.load(f)
    except:
        pass


    data.update(best_config_per_combo)
    with open(config_filepath, 'w') as f:
        json.dump(data, f, indent=4)

    return best_config_per_combo


config_path = "binary/configs/ComboGrid_gw6_withwalls.json"
analyze_auc(settings=["both", "internal", "input"], option_mode="didec", config_filepath=config_path)
analyze_auc(settings=["both"], option_mode="neural-augmented", config_filepath=config_path)
analyze_auc(settings=["both"], option_mode="dec-whole", config_filepath=config_path)
analyze_auc(settings=["both"], option_mode="vanilla", config_filepath=config_path)
analyze_auc(settings=["both"], option_mode="fine-tune", config_filepath=config_path)