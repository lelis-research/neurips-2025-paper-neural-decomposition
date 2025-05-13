import os
import shutil
import sys
import torch
import gymnasium as gym
import tyro
import random
import numpy as np
from dataclasses import dataclass
sys.path.append("/home/iprnb/scratch/neurips-2025-paper-neural-decomposition")
sys.path.append("C:\\Users\\Parnian\\Projects\\neurips-2025-paper-neural-decomposition")
from environments.environments_combogrid import PROBLEM_NAMES
from agents.recurrent_agent import GruAgent
from pipelines.option_discovery import get_single_environment
from environments.environments_minigrid import get_simplecross_env, make_env_simple_crossing, get_unlock_env
from utils import utils

@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppoAgent"
    """the name of this experiment"""
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, SimpleCrossing, FourRooms, Unlock, MultiRoom]
    """

    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'.
    This determines the exact environments that will be separately used for training.
    """
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    # hyperparameter arguments
    game_width: int = 4
    """the length of the combo/mini-grid square"""
    max_episode_length: int = 35
    """"""
    visitation_bonus: int = 1
    """"""
    use_options: int = 0
    """"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3
    """"""
    view_size: int = 5
    """the size of the agent's view in the mini-grid environment"""
    save_run_info: int = 0
    """save entropy and episode length along with satate dict if set to 1"""

 
    env_seed: int = 12
    """the seed of the environment (set in runtime)"""
    seed: int = 2
    """experiment randomness seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""



def find_pytorch_models(directory):
    pytorch_models = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):
                pytorch_models.append(os.path.join(root, file))
    return pytorch_models

def search_within_seeds(args, logger, directory_paths):
    best_models = []
    useful_seeds_per_env = {}
    for directory_path in directory_paths:
        models = find_pytorch_models(directory_path)
        logger.info(f"Found {len(models)} PyTorch models in {directory_path}.")
        env_seed = int(directory_path.strip().split("_")[-1])
        models_per_seed = {}
        useful_seeds_per_env[env_seed] = set()

        for model in models:
            # if args.env_id == "ComboGrid" and "actor" not in model:
            #     continue
            init_index = model.find("seed")
            seed = int(model[init_index:][len("seed")+1:model[init_index:].find(os.sep)])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic   
            if args.env_id == "ComboGrid":
                env = get_single_environment(seed=env_seed, args=args)
            elif args.env_id == "SimpleCrossing":
                env = get_single_environment(seed=env_seed, args=args)
            elif args.env_id == "Unlock":
                env = get_unlock_env(seed=env_seed, view_size=3, n_discrete_actions=5, args=args)
            try:
                checkpoint = torch.load(model, weights_only=True)
            except:
                # print(model)
                checkpoint = torch.load(os.path.join(directory_path,f"seed={seed}",model[:model.find("-ent_an0")] + ".pt"), weights_only=True)
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
            length_cap = args.max_episode_length
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
                print(f"MODEL {model} env {env_seed} seed {seed} length {current_length}")
                useful_seeds_per_env[env_seed].add(seed)
    print(useful_seeds_per_env[0].intersection(useful_seeds_per_env[1], useful_seeds_per_env[2], useful_seeds_per_env[3]))

def find_hyperparam_set(args, logger, directory_paths):
    best_models = []
    
    for directory_path in directory_paths:
        models = find_pytorch_models(directory_path)
        logger.info(f"Found {len(models)} PyTorch models in {directory_path}.")
        env_seed = int(directory_path.strip().split("_")[-1])
        models_per_seed = {}

        for model in models:
            if args.env_id == "ComboGrid" and "actor" not in model:
                continue
            init_index = model.find("seed")
            seed = int(model[init_index:][len("seed")+1:model[init_index:].find(os.sep)])
            if seed > 3: continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic   

            if args.env_id == "ComboGrid" and model.find("ent_an") == -1:
                model = model[:model.find(".pt")] + "-ent_an0.pt"
            if seed not in models_per_seed:
                models_per_seed[seed] = []
            models_per_seed[seed].append(model.split(os.path.sep)[-1])
        
        seeds = list(models_per_seed.keys())
        
        best_model = None
        best_model_length = float("inf")
        for model in models_per_seed[seeds[0]]:
            model_return_sum = 0
            considering = True
            for seed in seeds:
                if model not in models_per_seed[seed]:
                    print(f'model {model} not in seed {seed}')
                    considering = False
                    break
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = args.torch_deterministic   
                if args.env_id == "ComboGrid":
                    env = get_single_environment(seed=env_seed, args=args)
                elif args.env_id == "SimpleCrossing":
                    env = get_single_environment(seed=env_seed, args=args)
                elif args.env_id == "Unlock":
                    env = get_unlock_env(seed=env_seed, view_size=3, n_discrete_actions=5, args=args)
                try:
                    checkpoint = torch.load(os.path.join(directory_path,f"seed={seed}",model), weights_only=True)
                except:
                    # print(model)
                    checkpoint = torch.load(os.path.join(directory_path,f"seed={seed}",model[:model.find("-ent_an0")] + ".pt"), weights_only=True)
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
                length_cap = args.max_episode_length
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
                    model_return_sum += current_length
                    print(f"model {model} seed {seed} rnv seed {env_seed} length {current_length}")
                else:
                    # print(f"model {model} seed {seed} not able to find the goal. Skipping this setting")
                    considering = False
                    break
            if considering:
                # print("Considering...")
                if model_return_sum < best_model_length:
                    best_model = model
                    best_model_length = model_return_sum
        print(f"Best model for env {args.env_id} and env seed {env_seed} is {best_model} with length {best_model_length}")


def main():
    args = tyro.cli(Args)
    args.exp_id = f"{args.env_id}_sweep"
    log_path = os.path.join(args.log_path, args.exp_id, "test_primary_models")
    logger, _ = utils.get_logger('sweep_selector_logger', args.log_level, log_path)
    directory_paths = []
    for x in os.listdir('binary'):
        if x.startswith(f'models_sweep_{args.env_id}') and os.path.isdir(os.path.join('binary', x)):
            directory_paths.append(os.path.join('binary', x))
       
    find_hyperparam_set(args, logger, directory_paths)
    # search_within_seeds(args, logger, directory_paths)


if __name__ == "__main__":
    main()
