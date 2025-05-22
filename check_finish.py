import os
import argparse

def check_folders(root_dir):
    """
    Scan each subdirectory of root_dir and report whether 'final.pt' exists.

    Args:
        root_dir (str): Path to the directory containing subfolders to check.

    Prints:
        A list of folders that have final.pt and those that do not.
    """
    has_file = []
    missing_file = []

    # iterate over entries in root_dir
    for entry in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, entry)
        if os.path.isdir(folder_path):
            target = os.path.join(folder_path, 'final.pt')
            if os.path.isfile(target):
                has_file.append(entry)
            else:
                missing_file.append(entry)

    # print results
    print(f"Folders containing 'final.pt' ({len(has_file)}):")
    for name in has_file:
        print(f"  {name}")

    print(f"\nFolders missing 'final.pt' ({len(missing_file)}):")
    for name in missing_file:
        print(f"  {name}")

if __name__ == '__main__':
    root = "Results_car_nstepdqn_best"
    check_folders(root)
