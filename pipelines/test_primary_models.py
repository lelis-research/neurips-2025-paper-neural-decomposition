import os
import shutil
import sys
import torch
import gymnasium as gym
import tyro
import random
import numpy as np
sys.path.append("/home/iprnb/scratch/neurips-2025-paper-neural-decomposition")
sys.path.append("C:\\Users\\Parnian\\Projects\\neurips-2025-paper-neural-decomposition")
from environments.environments_combogrid import PROBLEM_NAMES
from agents.recurrent_agent import GruAgent
from pipelines.option_discovery import get_single_environment
from environments.environments_minigrid import get_simplecross_env
from pipelines.train_ppo import Args
from utils import utils




def find_pytorch_models(directory):
    pytorch_models = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pytorch_models.append(os.path.join(root, file))
    return pytorch_models

def main():
    args = tyro.cli(Args)
    args.exp_id = f"{args.env_id}_sweep"
    log_path = os.path.join(args.log_path, args.exp_id, "test_primary_models")
    logger, _ = utils.get_logger('sweep_selector_logger', args.log_level, log_path)
    directory_paths = []
    for x in os.listdir('binary'):
        if x.startswith(f'models_sweep_{args.env_id}') and os.path.isdir(os.path.join('binary', x)):
            directory_paths.append(os.path.join('binary', x))

    best_models = []
    for directory_path in directory_paths:
        models = find_pytorch_models(directory_path)
        logger.info(f"Found {len(models)} PyTorch models in {directory_path}.")
        # env_seed = int(directory_path.strip().split("_")[-1])
        env_seed = 3
        best_model = None
        best_seed = None
        best_length = float('inf')
        for model in models:
            #set seed for reproducibility
            init_index = model.find("seed")
            seed = int(model[init_index:][len("seed")+1:model[init_index:].find(os.sep)])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic

            #load agent
            if args.env_id == "ComboGrid":
                env = get_single_environment(seed=env_seed, args=args)
            elif args.env_id == "SimpleCrossing":
                env = get_simplecross_env(seed=env_seed, view_size=args.view_size, max_episode_steps=args.max_episode_length)
            agent = GruAgent(env, h_size=args.hidden_size)
            agent.load_state_dict(torch.load(model, weights_only=True))
            agent.eval()

            #evaluate agent
            next_rnn_state = agent.init_hidden()
            next_done = torch.zeros(1)
            o, _ = env.reset()
            length_cap = 100
            current_length = 0
            done = False

            while not done:
                o = torch.tensor(o, dtype=torch.float32)
                a, _, _, _, next_rnn_state, _ = agent.get_action_and_value(o, next_rnn_state, next_done)
                next_o, _, terminal, truncated, _ = env.step(a.item())
                current_length += 1
                if (length_cap is not None and current_length > length_cap) or \
                    terminal or truncated:
                    done = True
                o = next_o  
            if terminal:
                if best_model is None or current_length < best_length:
                    best_model = model
                    best_length = current_length
                    best_seed = seed
        if best_model is not None:
            logger.info(f"Best model for seed {best_seed} and env {PROBLEM_NAMES[env_seed] if args.env_id == "ComboGrid" else env_seed} is {best_model} with length {best_length}.")
            best_models.append((best_model, env_seed, best_seed))

    separator = os.path.sep
    # Copy the best models to the binary/models directory
    for best_model, env_seed, seed in best_models:
        source_file = best_model
        destination_file = f'binary{separator}models{separator}{args.env_id}{separator}seed={seed}{separator}{args.env_id.lower()}-{PROBLEM_NAMES[env_seed] if args.env_id == "ComboGrid" else env_seed}-{seed}.pt'
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        shutil.copy(source_file, destination_file)
        logger.info(f"Copied {source_file} to {destination_file}.")



if __name__ == "__main__":
    main()
