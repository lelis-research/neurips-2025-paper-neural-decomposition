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
from environments.environments_combogrid import PROBLEM_NAMES, SEEDS
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
    game_width: int = 3
    """the length of the combo/mini-grid square"""
    max_episode_length: int = 30
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


def check_seeds(args, logger, directory_path):
    best_models = []
    

    models = find_pytorch_models(directory_path)
    logger.info(f"Found {len(models)} PyTorch models in {directory_path}.")
    no_good_seeds = set()
    for model in models:
        env_seed = SEEDS[model.strip().split("-")[-3] + "-" + model.strip().split("-")[-2]]
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
            print(model)
            # checkpoint = torch.load(os.path.join(directory_path,f"seed={seed}",model[:model.find("-ent_an0")] + ".pt"), weights_only=True)
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
            pass 
            # print(f"model {model} seed {seed} rnv seed {env_seed} length {current_length}")
        else:
            if env_seed != 0:
                no_good_seeds.add(seed)
                print(f"IGNORE SEED {seed} ENV SEED {env_seed}")
    print(no_good_seeds)




def main():
    args = tyro.cli(Args)
    args.exp_id = f"{args.env_id}_sweep"
    log_path = os.path.join(args.log_path, args.exp_id, "test_primary_models")
    logger, _ = utils.get_logger('sweep_selector_logger', args.log_level, log_path)
    directory_path = f"binary/models/{args.env_id}"
    # if args.env_id == "ComboGrid":
    #     directory_path = directory_path + f"/width={args.game_width}"
    models = check_seeds(args, logger, directory_path)



if __name__ == "__main__":
    main()
