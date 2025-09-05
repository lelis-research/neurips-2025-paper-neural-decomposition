#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

# ======== CONFIG ========
BASE_DIR   = Path(".")          # where to search
SEEDS      = [2, 6, 8, 12, 44, 14, 46, 20, 55, 25, 28]  # <-- put your seeds in order
ENV_INDEX  = 3              # e.g., 0 to only process env_0; or None for all envs
STEPS      = "1000000"        # e.g., "150000" to only process that step count; or "all" for all steps
DRY_RUN    = False               # set to False to actually rename/delete
OFFSET    = 27                 # seed offset in folder names (e.g., 4 if seeds start from 4)
NORMALIZATION_OFFSET = 1000
# ========================

PATTERN = re.compile(r"^ComboGrid4_(\d+)_(\d+)_env_(\d+)$")

def seed_to_index(seed: int) -> int:
    try:
        return SEEDS.index(seed) + OFFSET + NORMALIZATION_OFFSET
    except ValueError:
        return -1

def main():
    # Find candidate directories
    candidates = []
    for p in BASE_DIR.iterdir():
        if p.is_dir():
            m = PATTERN.match(p.name)
            if m:
                seed = int(m.group(1))
                steps = m.group(2)
                env = int(m.group(3))
                if (ENV_INDEX is None or env == ENV_INDEX) and steps == STEPS:
                    candidates.append((p, seed, steps, env))

    if not candidates:
        print("No matching folders found.")
        return

    print(f"Found {len(candidates)} matching folders.")
    for p, seed, steps, env in sorted(candidates, key=lambda x: (x[3], x[1], x[2])):
        idx = seed_to_index(seed)
        if idx >= 0:
            # Rename to seed index
            new_name = f"ComboGrid4_{idx}_{steps}_env_{env}"
            new_path = p.parent / new_name
            if new_path.exists():
                if new_path.resolve() == p.resolve():
                    print(f"[SKIP] Already correct: {p.name}")
                    continue
                print(f"[WARN] Destination exists, skipping: {new_path}")
                continue
            print(f"[RENAME] {p.name}  ->  {new_name}")
            if not DRY_RUN:
                p.rename(new_path)
        else:
            # Not in seed list
            if seed < 60:
                print(f"[REMOVE] {p.name} (seed {seed} not in list and < 60)")
                if not DRY_RUN:
                    shutil.rmtree(p)
            else:
                print(f"[KEEP]   {p.name} (seed {seed} >= 60)")

    if DRY_RUN:
        print("\nNOTE: DRY_RUN is ON. No changes were made. Set DRY_RUN=False to apply.")

if __name__ == "__main__":
    main()
