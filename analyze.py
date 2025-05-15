import os
import torch

def compare_models_l2(model1, model2):
    total_diff = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        total_diff += torch.norm(p1 - p2).item()
    return total_diff



option_dir_g = os.path.join("Results", "Options_Mask_Maze_m_Seed_60000", "selected_options_5.pt")
option_dir_b = os.path.join("Results", "Options_Mask_Maze_m_Seed_220000", "selected_options_5.pt")

best_options_g = torch.load(option_dir_g, weights_only=False)
best_options_b = torch.load(option_dir_b, weights_only=False)

for i in range(len(best_options_b)):
    option_g = best_options_g[i]
    option_b = best_options_b[i]
    print(option_g.max_len, option_b.max_len)
    print(option_g.mask, option_b.mask)

