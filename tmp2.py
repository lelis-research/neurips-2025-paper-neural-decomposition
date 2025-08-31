import os
import re
from pipelines.option_discovery import regenerate_trajectories, process_args
from pipelines.option_discovery import Args
import tyro

def find_seeds_with_four_pt_files(root_dir):
    args = process_args()
    seeds_with_four = []

    # Regex to match folder names like "seed=123"
    seed_pattern = re.compile(r"seed=(\d+)")

    for entry in os.scandir(root_dir):
        if entry.is_dir():
            match = seed_pattern.match(entry.name)
            if match:
                seed_num = int(match.group(1))
                args.seed = seed_num
                # Count .pt files in this folder
                pt_files = [f for f in os.listdir(entry.path) if f.endswith("combo4.pt")]
                try:
                    regenerate_trajectories(args)
                    seeds_with_four.append(seed_num)
                except:
                    pass
    return seeds_with_four



if __name__ == "__main__":
    root_directory = "binary/models/MiniGrid-Unlock-v0_width=9_vanilla"  # Change this to the path where your seed folders live
    result = find_seeds_with_four_pt_files(root_directory)
    result.sort()
    print("Seeds with exactly 4 .pt files:", result, len(result))
