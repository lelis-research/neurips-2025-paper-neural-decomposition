import copy
import os
import random
import traceback
import torch
import tyro
import pickle
import glob
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List
from utils import utils
from dataclasses import dataclass
import itertools
import logging
from typing import Union, List, Tuple
import concurrent.futures
from pipelines.losses import LevinLossActorCritic, LogitsLossActorCritic
from agents.policy_guided_agent import PPOAgent
from environments.environments_combogrid_gym import ComboGym
from environments.environments_combogrid import SEEDS, PROBLEM_NAMES as COMBO_PROBLEM_NAMES
from environments.environments_minigrid import get_training_tasks_simplecross
from environments.utils import get_single_environment
from utils.utils import timing_decorator
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class Args:
    exp_name: str = "extract_basePolicyTransferred"
    """the name of this experiment"""
    # env_seeds: Union[List, str, Tuple] = (0,1,2,3)
    # env_seeds: Union[List, str, Tuple] = (0,1,2)
    env_seeds: Union[List, str, Tuple] = (1,3,17)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    # model_paths: List[str] = (
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    # )

    # model_paths: List[str] = (
    #     'minigrid-simplecrossings9n1-v0-0',
    #     'minigrid-simplecrossings9n1-v0-1',
    #     'minigrid-simplecrossings9n1-v0-2'
    # )
    model_paths: List[str] = (
            'minigrid-unlock-v0-1-3',
            'minigrid-unlock-v0-3-3',
            'minigrid-unlock-v0-17-3'
        )

    # These attributes will be filled in the runtime
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    problems: List[str] = ()
    """the name of the problems the agents were trained on; To be filled in runtime"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-Unlock-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameters
    game_width: int = 9
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    mask_transform_type: str = "softmax"
    option_mode: str = "neural-augmented"

    # Script arguments
    seed: int = 0
    """The seed used for reproducibilty of the script"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


def process_args() -> Args:
    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}' + \
        f'_envsd{",".join(map(str, args.env_seeds))}'

    # updating log path
    args.log_path = os.path.join(args.log_path, args.exp_id, f"seed={str(args.seed)}")
    
    # Processing seeds from commands
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        start, end = map(int, args.env_seeds.split(","))
        args.env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        args.problems = [COMBO_PROBLEM_NAMES[seed] for seed in args.env_seeds]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0" or "MiniGrid-Unlock-v0":
        args.problems = [args.env_id + f"_{seed}" for seed in args.env_seeds]
        
    return args


def regenerate_trajectories(args: Args, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{args.env_id}_width={args.game_width}_vanilla/seed={args.seed}/{model_directory}.pt'
        env = get_single_environment(args, seed=seed)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory, _ = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")

    return trajectories


def save_options(options: List[PPOAgent], trajectories: dict, args: Args, logger):
    """
    Save the options (masks, models, and number of iterations) to the specified directory.

    Parameters:
        options (List[PPOAgent]): The models corresponding to the masks.
        trajectories (Dict[str, Trajectory]): The trajectories corresponding to the these options
        save_dir (str): The directory where the options will be saved.
    """
    save_dir = f"binary/options/{args.env_id}_width={args.game_width}_{args.option_mode}/seed={args.seed}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Removing every file or directory in the path 
        for file_path in glob.glob(os.path.join(save_dir, "*")):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)

    trajectories_path = os.path.join(save_dir, f'trajectories.pickle')
    with open(trajectories_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    logger.info(f"Trajectories saved to {trajectories_path}")
    
    # Save each model with its mask and iteration count
    for i, model in enumerate(options):
        
        model_path = os.path.join(save_dir, f'ppo_model_option_{i}.pth')
        torch.save({
            'id': i,
            'model_state_dict': model.state_dict(),
            'mask': model.mask,
            'n_iterations': model.option_size,
            'problem': model.problem_id,
            'environment_args': vars(args),
            'extra_info': model.extra_info
        }, model_path)
    
    logger.info(f"Options saved to {save_dir}")


def load_options(args, logger):
    """
    Load the saved options (masks, models, and number of iterations) from the specified directory.

    Parameters:
        save_dir (str): The directory where the options, and trajectories are saved.

    Returns:
        options (List[PPOAgent]): Loaded models.
        loaded_trajectories (List[Trajectory]): Loaded trajectories.
    """

    # Load the models and iterations
    save_dir = f"binary/options/{args.exp_id}/seed={args.seed}"

    logger.info(f"Option directory: {save_dir}")

    model_files = sorted([f for f in os.listdir(save_dir) if f.startswith('ppo_model_option_') and f.endswith('.pth')])
    
    logger.info(f"Found options: {model_files}")

    n = len(model_files)
    options = [None] * n

    print(model_files)
    

    for model_file in model_files:
        model_path = os.path.join(save_dir, model_file)
        checkpoint = torch.load(model_path)
        assert int(checkpoint['environment_args']['game_width']) == int(args.game_width)
        
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            if 'environment_args' in checkpoint:
                seed = int(checkpoint['environment_args']['seed'])
            else:
                seed = int(checkpoint['problem'][-1])
        elif args.env_id == "ComboGrid":
            seed = None
            problem = checkpoint['problem']
        else:
            raise NotImplementedError
        envs = get_single_environment(args, seed, problem) # adding options and test/train configuration is not necessary

        model = PPOAgent(envs=envs, hidden_size=args.hidden_size)  # Create a new PPOAgent instance with default parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to_option(checkpoint['mask'], checkpoint['n_iterations'], checkpoint['problem'])
        model.extra_info = checkpoint['extra_info'] if 'extra_info' in checkpoint else {}
        model.environment_args = checkpoint['environment_args'] if 'environment_args' in checkpoint else {}

        i = checkpoint['id']
        options[i] = model
        
    trajectories_path = os.path.join(save_dir, f'trajectories.pickle')

    with open(trajectories_path, 'rb') as f:
        loaded_trajectory = pickle.load(f)

    return options, loaded_trajectory


class WholeDecOption:
    def __init__(self, args: Args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.levin_loss = LevinLossActorCritic(self.logger, mask_transform_type=self.args.mask_transform_type)

    
    @timing_decorator
    def discover(self):
        """
        """
        

        trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)
        
        
        selected_options = []
        for primary_env_seed, primary_problem, primary_model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
            env = get_single_environment(self.args, seed=primary_env_seed)
            option = PPOAgent(env, hidden_size=self.args.hidden_size)
            option.load_state_dict(torch.load(os.path.join("binary/models", f"{self.args.env_id}_width={self.args.game_width}_vanilla" ,f"seed={self.args.seed}", f"{primary_model_directory}.pt")))
            mask = torch.zeros(3, self.args.hidden_size)
            mask[-1] = 1
            option.to_option(mask, 1, primary_problem)  
            selected_options.append(option)

        # printing selected options
        self.logger.info("Selected options:")
        for i in range(len(selected_options)):
            self.logger.info(f"Option #{i}:\n" + 
                        f"size={selected_options[i].option_size}\n" +
                        f"extra_info={selected_options[i].extra_info}\n" )

        save_options(options=selected_options, 
                    trajectories=trajectories,
                    args=self.args, 
                    logger=self.logger)

        utils.logger_flush(self.logger)

        self.levin_loss.print_output_subpolicy_trajectory(selected_options, trajectories, logger=self.logger)
        utils.logger_flush(self.logger)

        self.logger.info("Testing on each grid cell")
        for seed, problem in zip(self.args.env_seeds, self.args.problems):
            self.logger.info(f"Testing on each cell..., {problem}")
            self.levin_loss.evaluate_on_each_cell(options=selected_options, 
                                    trajectories=trajectories,
                                    problem_test=problem, 
                                    args=self.args, 
                                    seed=seed, 
                                    logger=self.logger)

        utils.logger_flush(self.logger)



def main():
    args = process_args()

    # Logger configurations
    logger, args.log_path = utils.get_logger(args.exp_name, args.log_level, args.log_path)

    run_name = f'{args.exp_id}_sd{args.seed}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_id,
            job_type="eval",
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)


    module_extractor = WholeDecOption(args, logger)
    module_extractor.discover()

    logger.info(f"Run id: {run_name}")
    logger.info(f"logs saved at {args.log_path}")


if __name__ == "__main__":
    main()
