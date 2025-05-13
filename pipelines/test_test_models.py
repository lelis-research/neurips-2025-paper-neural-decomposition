import os
import shutil
import sys
import torch
import gymnasium as gym
import tyro
import random
import numpy as np
import pandas as pd
sys.path.append("/home/iprnb/scratch/neurips-2025-paper-neural-decomposition")
sys.path.append("C:\\Users\\Parnian\\Projects\\neurips-2025-paper-neural-decomposition")
from environments.environments_combogrid import PROBLEM_NAMES
from agents.recurrent_agent import GruAgent
from pipelines.option_discovery import get_single_environment, load_options
from environments.environments_minigrid import get_simplecross_env, make_env_simple_crossing, get_unlock_env, get_fourrooms_env
from pipelines.train_ppo import Args
from utils import utils



entropy_threshold = 0.1  # low entropy means confident policy
length_threshold = 50    # goal reached in ~25 steps
sustain_steps = 10        # number of consecutive points required to consider it stable

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
    log_path = os.path.join(args.log_path, args.exp_id, "test_test_models")
    logger, _ = utils.get_logger('sweep_selector_logger', args.log_level, log_path)
    directory_paths = []
    for x in os.listdir('binary'):
        if x.startswith(f'models_sweep_{args.env_id}') and os.path.isdir(os.path.join('binary', x)):
            directory_paths.append(os.path.join('binary', x))
    
    if args.env_id == "FourRooms":
            option_folder = f"selected_options/SimpleCrossing"
    elif args.env_id == "ComboGrid":
        option_folder = f"selected_options/ComboGrid"
    options, _ = load_options(args, logger, folder=option_folder)

    best_models = []
    for directory_path in directory_paths:
        models = find_pytorch_models(directory_path)
        logger.info(f"Found {len(models)} PyTorch models in {directory_path}.")
        env_seed = int(directory_path.strip().split("_")[-1])
        best_model_per_seed_no_option = {} 
        best_model_per_seed_with_option = {}
        
        best_seed = None
        best_length = float('inf')
        for model in models:
            #set seed for reproducibility
            init_index = model.find("seed")
            seed = int(model[init_index:][len("seed")+1:model[init_index:].find(os.sep)])
            use_options = int(model[model.find("option")+len("option")])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            args.seed = seeds
            options, _ = load_options(args, logger, folder=option_folder)
            torch.backends.cudnn.deterministic = args.torch_deterministic

            #load agent
            if args.env_id == "ComboGrid":
                if use_options == 1:
                    env = get_single_environment(seed=env_seed, args=args, options=options)
                else:
                    env = get_single_environment(seed=env_seed, args=args)
            elif args.env_id == "SimpleCrossing":
                env = get_single_environment(seed=env_seed, args=args)
            elif args.env_id == "Unlock":
                env = get_unlock_env(seed=env_seed, view_size=3, n_discrete_actions=5, args=args)
            elif args.env_id == "FourRooms":
                env = get_fourrooms_env(seed=env_seed, view_size=5, args=args, options=options if use_options == 1 else None)
                
            
            checkpoint = torch.load(model, weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                episode_lengths = checkpoint['episode_lengths']
                entropies = checkpoint['policy_entropies']
                steps = ['steps']
            else:
                state_dict = checkpoint
            
            agent = GruAgent(env, h_size=args.hidden_size, env_id=args.env_id)
            agent.load_state_dict(state_dict)
            agent.eval()

            #evaluate agent
            next_rnn_state = agent.init_hidden()
            next_done = torch.zeros(1)
            o, _ = env.reset()
            length_cap = 70
            current_length = 0

            while True:
                o = torch.tensor(o, dtype=torch.float32)
                a, _, _, _, next_rnn_state, _ = agent.get_action_and_value(o, next_rnn_state, next_done)
                next_o, _, terminal, truncated, _ = env.step(a.item())
                o = next_o  
                current_length += 1
                # print("Step: ", current_length, "Action: ", a.item())
                if terminal or truncated or current_length > length_cap:
                    break
            if terminal:
                if use_options == 1:
                    best_model_per_seed = best_model_per_seed_with_option
                else:   
                    best_model_per_seed = best_model_per_seed_no_option
                if seed not in best_model_per_seed: 
                    best_model_per_seed[seed] = {"length": float("inf"), 'model': None}
                #Calculate the approximate step where the model converged to an acceptable policy
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        if "convergence_step" not in best_model_per_seed[seed]:
                            best_model_per_seed[seed]['convergence_step'] = float("inf")
                        print(len(episode_lengths), len(entropies), len(steps))
                        log_data = pd.DataFrame({
                        'episode_length': episode_lengths,
                        'policy_entropy': entropies
                    })
                        converged = (
                    (log_data['policy_entropy'] < entropy_threshold) &
                    (log_data['episode_length'] < length_threshold)
                    )
                        rolling_sum = converged.rolling(window=sustain_steps).sum()
                        rolling_sum[rolling_sum == sustain_steps].first_valid_index()
                        # Find first step where the window shows sustained convergence
                        first_converged_index = rolling_sum[rolling_sum == sustain_steps].first_valid_index()
                        if first_converged_index is not None:
                            convergence_step = first_converged_index
                        else:
                            convergence_step = float("inf")
                        print(f"Convergence step is {convergence_step}.")
                        
                        if best_model_per_seed[seed]['model'] is None or (current_length <= best_model_per_seed[seed]['length'] and convergence_step <= best_model_per_seed[seed]['convergence_step']):
                            best_model_per_seed[seed]['model'] = model
                            best_model_per_seed[seed]['length'] = current_length
                            best_model_per_seed[seed]['convergence_step'] = convergence_step
                else:            
                    if best_model_per_seed[seed]['model'] is None or current_length < best_model_per_seed[seed]['length']:
                        best_model_per_seed[seed]['model'] = model
                        best_model_per_seed[seed]['length'] = current_length
        if len(best_model_per_seed_no_option) != 0:
            for best_seed in best_model_per_seed:
                logger.info(f"Best model for seed {best_seed} and env {PROBLEM_NAMES[env_seed] if args.env_id == 'ComboGrid' else env_seed} is {best_model_per_seed_no_option[best_seed]['model']} with length {best_model_per_seed_no_option[best_seed]['length']} \
                            Convergence rate is {best_model_per_seed_no_option[best_seed]['convergence_step'] if 'convergence_step' in best_model_per_seed_no_option[best_seed] else 'NA'}.")
                
                best_models.append((best_model_per_seed_no_option[best_seed]['model'], env_seed, best_seed, 0))
        if len(best_model_per_seed_with_option) != 0:
            for best_seed in best_model_per_seed_with_option:
                logger.info(f"Best model for seed {best_seed} and env {PROBLEM_NAMES[env_seed] if args.env_id == 'ComboGrid' else env_seed} is {best_model_per_seed_with_option[best_seed]['model']} with length {best_model_per_seed_with_option[best_seed]['length']} \
                            Convergence rate is {best_model_per_seed_with_option[best_seed]['convergence_step'] if 'convergence_step' in best_model_per_seed_with_option[best_seed] else 'NA'}.")
                
                best_models.append((best_model_per_seed_with_option[best_seed]['model'], env_seed, best_seed, 1))

    separator = os.path.sep
    # Copy the best models to the binary/models directory
    for best_model, env_seed, seed, use_options in best_models:
        source_file = best_model
        destination_file = f'binary{separator}models{separator}{args.env_id}{separator}seed={seed}{separator}{args.env_id.lower()}-{PROBLEM_NAMES[env_seed] if args.env_id == "ComboGrid" else env_seed}options{use_options}-{seed}.pt'
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        shutil.copy(source_file, destination_file)
        logger.info(f"Copied {source_file} to {destination_file}.")



if __name__ == "__main__":
    main()
