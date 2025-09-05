import os
import re
import sys

# Count a line as "reward == 1.0" if ANY return on that line equals 1.0
# Example line:
#   Episode: 6688 Timestep: 286896 Return Org: 1.00 Return Wrapped: 1.00
RETURN_RE = re.compile(r"Return\s+(?:Org|Wrapped):\s*([0-9]+(?:\.[0-9]+)?)")

def has_consecutive_ones(path, needed=50):
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                vals = [float(m.group(1)) for m in RETURN_RE.finditer(line)]
                if any(abs(v - 1.0) < 1e-9 for v in vals):
                    count += 1
                    if count >= needed:
                        return True
                else:
                    count = 0
    except Exception:
        # Treat unreadable files as "bad"
        return False
    return False

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    needed = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".out"):
                path = os.path.join(dirpath, name)
                if not has_consecutive_ones(path, needed):
                    print(path)

if __name__ == "__main__":
    main()
