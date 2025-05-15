import os
import re
import subprocess
import getpass

# --- Wandb login ---
print("🔑 Logging into wandb...")
wandb_token = "af92d00a13698da89f8ff2ae5e2d8bc4d932e26a"
final_command = f"export WANDB_API_KEY=af92d00a13698da89f8ff2ae5e2d8bc4d932e26a && wandb login {wandb_token}"
try:
    subprocess.run(f"export WANDB_API_KEY=af92d00a13698da89f8ff2ae5e2d8bc4d932e26a && wandb login {wandb_token}", shell=True, check=True)
    print("✅ wandb login successful")
except subprocess.CalledProcessError as e:
    print(f"❌ wandb login failed: {e}")
    exit(1)

# --- Pattern to match wandb sync line ---
wandb_sync_pattern = re.compile(r"wandb:\s*(wandb sync\s+.+)")
start = 0
# start = 305918
# end = 304774
end = 1_000_000
jobid_pattern = re.compile(r"(\d+)-*")

path = "/home/rezaabdz/projects/aip-lelis/rezaabdz/neurips-2025-paper-neural-decomposition/experiments/testbasepolicytransfercombo12"
# --- Loop through .out files ---
print("files:", os.listdir(path))
for file in os.listdir(path):
    if not file.endswith(".out"):
        continue
    if file.endswith("copy.out"):
        continue

    if (match := jobid_pattern.search(file)):
        job_id = int(match.group(1))
    if job_id < start or job_id > end:
        continue

    with open(os.path.join(path, file), "r") as f:
        for line in f:
            match = wandb_sync_pattern.search(line)
            if match:
                command = match.group(1)
                final_command += "; " + command
                print(f"📤 Executing: {command}")
                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to run command in {file}: {e}")
                break  # Only execute the first wandb sync found per file

subprocess.run(final_command, shell=True, check=True)