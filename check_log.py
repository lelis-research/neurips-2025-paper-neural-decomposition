#!/usr/bin/env python3
import os
import re
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# how many bytes to read from the end of each file
DEFAULT_TAIL = 64 * 1024

# byte-markers & regexes
DONE_BYTES      = b"*** Num selected options:"
# old-style: "restarts=237/500"
RESTART_RE      = re.compile(rb"restarts=(?P<done>\d+)/(?P<total>\d+)")
# new-style: "Restarts: ... | 498/500 [34:38<00:07,"
PROG_RE         = re.compile(
    rb"Restarts:.*\|\s*(?P<done>\d+)/(?P<total>\d+)\s*\[.*?<(?P<eta>\d{1,2}:\d{2}(?::\d{2})?),"
)
ETA_RE          = re.compile(rb"<(?P<eta>\d{1,2}:\d{2}:\d{2}),")
SUFFIXES        = ["mask.out", "decwhole.out", "finetuning.out"]

def find_last_eta_full(path):
    """Fallback full scan for ETA stamps if needed."""
    last_eta = None
    with open(path, "rb") as f:
        for line in f:
            m = ETA_RE.search(line)
            if m:
                last_eta = m.group("eta").decode()
    return last_eta

def analyze_file(path, tail_size):
    """
    Read only the last `tail_size` bytes, then in priority:
      1) DONE_BYTES → "done"
      2) PROG_RE    → "restarts X/Y, ETA remaining HH:MM(:SS)"
      3) RESTART_RE → "restarts X/Y"
      4) ETA_RE     → "ETA remaining HH:MM:SS"
      5) fallback full scan for finetuning ETAs
      6) else → "no matching progress"
    Returns (exp_key, basename, status_str).
    """
    name = os.path.basename(path)
    # figure out experiment group
    exp = "other"
    for suf in SUFFIXES:
        if name.endswith(suf):
            exp = suf.rsplit(".", 1)[0]
            break

    size = os.path.getsize(path)
    read_bytes = min(size, tail_size)
    with open(path, "rb") as f:
        f.seek(-read_bytes, os.SEEK_END)
        tail = f.read()

    # 1) done?
    if DONE_BYTES in tail:
        status = "done"
    else:
        # 2) new progress bar
        m = PROG_RE.search(tail)
        if m:
            done = m.group("done").decode()
            tot  = m.group("total").decode()
            eta  = m.group("eta").decode()
            status = f"restarts {done}/{tot}, ETA remaining {eta}"
        else:
            # 3) old restart= style
            m2 = RESTART_RE.search(tail)
            if m2:
                done = m2.group("done").decode()
                tot  = m2.group("total").decode()
                status = f"restarts {done}/{tot}"
            else:
                # 4) any ETA in tail?
                matches = list(ETA_RE.finditer(tail))
                if matches:
                    status = f"ETA remaining {matches[-1].group('eta').decode()}"
                else:
                    status = "no matching progress"
                    # 5) fallback only for finetuning
                    if exp == "finetuning":
                        full_eta = find_last_eta_full(path)
                        if full_eta:
                            status = f"ETA remaining {full_eta}"

    return exp, name, status

def main(log_dir, workers, tail_size):
    # gather all seed* files
    seed_files = sorted(
        os.path.join(log_dir, fn)
        for fn in os.listdir(log_dir)
        if fn.startswith("seed")
    )
    total = len(seed_files)
    if total == 0:
        print("No files starting with 'seed' found.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze_file, p, tail_size): p for p in seed_files}
        for idx, fut in enumerate(as_completed(futures), start=1):
            exp, name, status = fut.result()
            print(f"[{idx}/{total}] scanned {name}")
            results.append((exp, name, status))

    # group & print sorted
    groups = defaultdict(list)
    for exp, name, status in results:
        groups[exp].append((name, status))

    exp_order = [s.rsplit(".",1)[0] for s in SUFFIXES] + ["other"]
    print("\n=== Summary by experiment ===")
    for exp in exp_order:
        if exp not in groups:
            continue
        print(f"\n--- {exp} ---")
        for name, status in sorted(groups[exp]):
            print(f"{name}: {status}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Constant-RAM, parallel scan of seed* logs, now handling the new Restarts:… format."
    )
    p.add_argument("log_dir", help="Directory containing your seed* log files")
    p.add_argument(
        "--workers", "-w", type=int, default=os.cpu_count(),
        help="Number of parallel workers (default: CPU count)"
    )
    p.add_argument(
        "--tail", "-t", type=int, default=DEFAULT_TAIL,
        help="Bytes to read from file tail (default: 64 KiB)"
    )
    args = p.parse_args()
    main(args.log_dir, args.workers, args.tail)