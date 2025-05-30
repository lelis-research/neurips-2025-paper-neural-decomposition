import os
import glob

# Parent folder containing your Options_* subfolders
root = "Results_MiniGrid_A2C_ReLU"

# Grab and sort all matching directories
dirs = sorted(glob.glob(os.path.join(root, "Options_*")))

for path in dirs:
    if not os.path.isdir(path):
        continue

    name = os.path.basename(path)
    missing = []

    # Determine required files based on prefix
    if name.startswith("Options_Transfer_"):
        required_files = ["selected_options.pt"]
    elif name.startswith("Options_FineTune") or name.startswith("Options_DecWhole") or name.startswith("Options_Mask"):
        required_files = [
            "trajectories.pt",
            "all_options.pt",
            "selected_options_5.pt",
            "selected_options_10.pt",
            "selected_options_20.pt"
        ]
    else:
        # Skip other prefixes
        continue

    # Check for missing files
    for fname in required_files:
        if not os.path.isfile(os.path.join(path, fname)):
            missing.append(fname)

    # Print only if there are missing files
    if missing:
        print(f"{name} is missing:")
        for fname in missing:
            print(f"    {fname}")
        print()  # Add a blank line for readability