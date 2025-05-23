import os
import re
import subprocess
import argparse

# --- CLI Argument Parser ---
parser = argparse.ArgumentParser(description="Sync wandb offline runs from .out files.")
parser.add_argument("path", type=str, help="Path to the directory containing .out files")
args = parser.parse_args()

path = args.path

# --- Wandb login ---
wandb_token = "af92d00a13698da89f8ff2ae5e2d8bc4d932e26a"
os.environ["WANDB_API_KEY"] = wandb_token
print("🔑 Logging into wandb...")

try:
    subprocess.run(f"wandb login {wandb_token}", shell=True, check=True)
    print("✅ wandb login successful")
except subprocess.CalledProcessError as e:
    print(f"❌ wandb login failed: {e}")
    exit(1)

# --- Regex patterns ---
wandb_sync_pattern = re.compile(r"wandb:\s*(wandb sync\s+.+)")
jobid_pattern = re.compile(r"(\d+)-*")

# --- Optional job range filter ---
start = 0
end = 1_000_000

# --- Process .out files ---
print("📁 Scanning directory:", path)
print("📄 Found files:", os.listdir(path))

for file in os.listdir(path):
    if not file.endswith(".out") or file.endswith("copy.out"):
        continue

    match = jobid_pattern.search(file)
    if match:
        job_id = int(match.group(1))
        if job_id < start or job_id > end:
            continue
    else:
        continue  # skip files without job ID

    with open(os.path.join(path, file), "r") as f:
        for line in f:
            match = wandb_sync_pattern.search(line)
            if match:
                command = match.group(1)
                print(f"📤 Executing: {command}")
                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to run command in {file}: {e}")
                break  # Only execute the first wandb sync found per file