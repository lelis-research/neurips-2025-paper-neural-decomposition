import os
import re

def extract_reward(line):
    """Extract the episodic_return value from a log line."""
    match = re.search(r"episodic_return=([0-9.]+)", line)
    if match:
        return float(match.group(1))
    return None

def check_out_files(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".out"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r") as f:
                    lines = f.readlines()

                # Filter lines that contain episodic_return
                reward_lines = [line for line in lines if "episodic_return=" in line]
                if not reward_lines:
                    continue  # No reward line found

                last_reward_line = reward_lines[-1]
                reward = extract_reward(last_reward_line)

                if reward is not None and reward != 1.0:
                    print(f"{filename} → episodic_return={reward}")

            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Run it on the current directory
if __name__ == "__main__":
    check_out_files(".")