#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from collections import defaultdict, namedtuple
import csv

# ---------- Config ----------
BASE_DIR = Path(".")  # change if needed
LOG_GLOB = "logs/*.out"
METRIC = "wrapped"

# Folder name pattern: steps{ts}_lr{lr}_clip{clip}_ent{ent}
FOLDER_RE = re.compile(
    r"^steps(?P<ts>\d+)_lr(?P<lr>[0-9.eE+-]+)_clip(?P<clip>[0-9.]+)_ent(?P<ent>[0-9.]+)$"
)

LINE_RE = re.compile(
    r"Episode:\s*(?P<ep>\d+)\s+Timestep:\s*(?P<t>\d+)\s+Return Org:\s*(?P<org>-?\d+(?:\.\d+)?)\s+Return Wrapped:\s*(?P<wrap>-?\d+(?:\.\d+)?)"
)

Combo = namedtuple("Combo", "ts lr clip ent folder")

def parse_folder_name(child: Path):
    m = FOLDER_RE.match(child.name)
    if not m:
        return None
    d = m.groupdict()
    combo = Combo(
        ts=int(d["ts"]),
        lr=float(d["lr"]),
        clip=float(d["clip"]),
        ent=float(d["ent"]),
        folder=child.name
    )
    return combo

def weighted_avg_from_log(path: Path):
    prev_t = 0
    weighted_sum = 0.0
    total_steps = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = LINE_RE.search(line)
                if not m:
                    continue
                t = int(m.group("t"))
                if t < prev_t:
                    prev_t = 0
                delta = t - prev_t
                prev_t = t
                if delta <= 0:
                    continue
                r = float(m.group("wrap")) if METRIC == "wrapped" else float(m.group("org"))
                weighted_sum += r * delta
                total_steps += delta
    except Exception:
        return None, 0
    if total_steps == 0:
        return None, 0
    return weighted_sum / total_steps, total_steps

def combine_weighted(avgs_and_steps):
    num = 0.0
    den = 0
    for avg, steps in avgs_and_steps:
        if avg is None or steps <= 0:
            continue
        num += avg * steps
        den += steps
    if den == 0:
        return None, 0
    return num / den, den

def main():
    combo_to_seed_stats = defaultdict(list)
    for child in BASE_DIR.iterdir():
        if not child.is_dir():
            continue
        combo = parse_folder_name(child)
        if not combo:
            continue
        logs = list(child.glob(LOG_GLOB))
        if not logs:
            continue
        file_stats = []
        for logf in logs:
            avg, steps = weighted_avg_from_log(logf)
            if steps > 0:
                file_stats.append((avg, steps))
        seed_avg, seed_steps = combine_weighted(file_stats)
        if seed_steps > 0:
            combo_to_seed_stats[combo].append((seed_avg, seed_steps))

    rows = []
    for combo, stats in combo_to_seed_stats.items():
        combo_avg, combo_steps = combine_weighted(stats)
        rows.append({
            "folder": combo.folder,
            "steps_config": combo.ts,
            "lr": combo.lr,
            "clip": combo.clip,
            "ent": combo.ent,
            "num_runs": len(stats),
            "total_steps": combo_steps,
            f"avg_return_{METRIC}": combo_avg if combo_avg is not None else float("nan"),
        })

    if not rows:
        print("No results found. Make sure your folders follow the expected naming and logs exist.")
        sys.exit(0)

    rows.sort(key=lambda r: (r[f"avg_return_{METRIC}"] if r[f"avg_return_{METRIC}"]==r[f"avg_return_{METRIC}"] else -1e18), reverse=True)

    header = ["folder","steps_config","lr","clip","ent","num_runs","total_steps",f"avg_return_{METRIC}"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r[k]) for k in header))

    best = rows[0]
    print("\nBest hyperparameters (by timestep-weighted average):")
    print(", ".join(f"{k}={best[k]}" for k in header))

    with open("sweep_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in header})

    print("\nSaved detailed summary to sweep_summary.csv")

if __name__ == "__main__":
    main()