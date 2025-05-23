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
import warnings
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class Args:
    exp_name: str = "extract_decOption"
    """the name of this experiment"""
    env_seeds: Union[List, str, Tuple] = (0,1,2,3)
    # env_seeds: Union[List, str, Tuple] = (0,1,2)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    # model_paths: List[str] = (
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    model_paths: List[str] = (
        'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
        'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
        'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
        'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
    )

    # These attributes will be filled in the runtime
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    problems: List[str] = ()
    """the name of the problems the agents were trained on; To be filled in runtime"""

    # Algorithm specific arguments
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameters
    game_width: int = 6
    """the length of the combo/mini grid square"""
    hidden_size: int = 6
    """"""

    mask_type: str = "internal"
    """It's one of these: [internal, input, both]"""
    mask_transform_type: str = "softmax"
    """It's either `softmax` or `quantize`"""
    selection_type: str = "greedy"
    """It's either `local_search` or `greedy`"""
    cache_path: str = ""
    """Path to the directory where the options are saved. If empty, it will be replaced based on the current `exp_id`"""
    max_num_options: int = 5

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
        f'_gw{args.game_width}_h{args.hidden_size}_envsd{",".join(map(str, args.env_seeds))}'
        if 'mask_type' in vars(args):
            args.exp_id += f'_mskType{args.mask_type}'
        if 'mask_transform_type' in vars(args):
            args.exp_id += f'_mskTransform{args.mask_transform_type}'
        if 'selection_type' in vars(args):
            args.exp_id += f'_selectType{args.selection_type}'

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
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        args.problems = [args.env_id + f"_{seed}" for seed in args.env_seeds]
        
    return args


def regenerate_trajectories(args: Args, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
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
    save_dir = f"binary/options/{args.exp_id}/seed={args.seed}"
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


class DecOption:
    def __init__(self, args: Args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.mask_transform_type = args.mask_transform_type
        self.mask_type = args.mask_type
        self.levin_loss = LevinLossActorCritic(self.logger, mask_type=self.mask_type, mask_transform_type=self.mask_transform_type)
        self.number_actions = 3

        self.selection_type = args.selection_type
        if args.cache_path == "":
            args.cache_path = "binary/cache/"
            self.option_candidates_path = os.path.join(args.cache_path, args.exp_id, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, args.exp_id, f"seed={args.seed}", "option_cache.pkl")
        else:
            self.option_candidates_path = os.path.join(args.cache_path, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"seed={args.seed}", "option_cache.pkl")

    @timing_decorator
    def discover(self):
        """
        This function performs hill climbing in the space of masks of a ReLU neural network
        to minimize the Levin loss of a given data set. It uses gumbel_softmax to extract 
        """

        trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)
        previous_loss = None
        best_loss = None

        selected_masks = []
        selected_models_of_masks = []
        selected_options_problem = []
        selected_number_iterations = []

        while previous_loss is None or best_loss < previous_loss:
            previous_loss = best_loss

            best_loss = None
            best_mask = None
            best_length = None
            model_best_mask = None
            problem_mask = None

            for seed, problem, model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
                self.logger.info(f'Extracting options using trajectory segments from {problem}, env_seed={seed}')
                model_file = f'binary/models/{model_directory}/seed={self.args.seed}/ppo_first_MODEL.pt'
                env = ComboGym(rows=self.args.game_width, columns=self.args.game_width, problem=problem)
                agent = PPOAgent(env, hidden_size=self.args.hidden_size)
                agent.load_state_dict(torch.load(model_file))

                t_length = len(trajectories[problem].get_state_sequence())

                mask, levin_loss, length = self.evaluate_all_masks_for_ppo_model(selected_masks, selected_models_of_masks, agent, problem, t_length, trajectories, 3, selected_number_iterations, self.args.hidden_size)

                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_mask = mask
                    best_length = length
                    model_best_mask = agent
                    problem_mask = problem
                    model_best_mask.to_option(best_mask, best_length, problem)

                    self.logger.info('Best Loss so far: ', best_loss, problem)

            # we recompute the Levin loss after the automaton is selected so that we can use 
            # the loss on all trajectories as the stopping condition for selecting automata
            selected_masks.append(best_mask)
            selected_models_of_masks.append(model_best_mask)
            selected_options_problem.append(problem_mask)
            selected_number_iterations.append(problem_mask)
            best_loss = self.levin_loss.compute_loss(selected_masks, selected_models_of_masks, "", trajectories, 3, selected_number_iterations)

            self.logger.info("Levin loss of the current set: ", best_loss)

        # remove the last automaton added
        num_options = len(selected_masks)
        selected_masks = selected_masks[0:num_options - 1]
        selected_models_of_masks = selected_models_of_masks[:num_options - 1]

        self.levin_loss.print_output_subpolicy_trajectory(selected_models_of_masks, selected_masks, selected_options_problem, trajectories, selected_number_iterations)

        selected_options = selected_models_of_masks

        # printing selected options
        self.logger.info("Selected options:")
        for i in range(len(selected_options)):
            self.logger.info(f"Option #{i}:\n" + 
                        f"mask={selected_options[i].mask}\n" +
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

    

    def evaluate_masks(self, combinations, masks, selected_models_of_masks, model, problem, max_option_size, trajectories, number_actions, number_iterations):
        best_value = None
        best_mask = None
        best_length = None
        for value in combinations:
            current_mask = torch.tensor(value, dtype=torch.long).view(1, -1)
            mapped_mask = (current_mask + 3) % 3

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                onehot_mask = F.one_hot(mapped_mask, num_classes=3).T

            for i in range(2, max_option_size):
            
                value = self.levin_loss.compute_loss(masks + [onehot_mask], selected_models_of_masks + [model], problem, trajectories, number_actions, number_iterations + [i])

                if best_mask is None or value < best_value:
                    best_value = value
                    best_mask = copy.deepcopy(onehot_mask)
                    best_length = i
        return best_value, best_mask, best_length

    def evaluate_all_masks_for_ppo_model(self, masks, selected_models_of_masks, model, problem, max_option_size, trajectories, number_actions, number_iterations, hidden_size):
        """
        Function that evaluates all masks for a given model. It returns the best mask (the one that minimizes the Levin loss)
        for the current set of selected masks. It also returns the Levin loss of the best mask. 
        """
        values = [-1, 0, 1]

        best_mask = None
        best_value = None
        best_length = None

        import math
        combinations = list(itertools.product(values, repeat=hidden_size))
        self.logger.info(f"total number of combinations: {len(combinations)}")
        batch_size = int(math.ceil(len(combinations) / self.args.cpus))
        combinations = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
        
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
        #     futures = []
        #     for batch in combinations:
        #         futures.append(executor.submit(self.evaluate_masks, batch, masks, selected_models_of_masks, model, problem, max_option_size, trajectories, number_actions, number_iterations))

        #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        #         try:
        #             value, mask, length = future.result()
        #             if best_mask is None or value < best_value:
        #                 best_value = value
        #                 best_mask = mask
        #                 best_length = length
        #         except Exception as e:
        #             traceback.print_exc()
        #             self.logger.error(f"Error in evaluating masks: {e}")
        #             return 
           
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
            futures = []
            for batch in combinations:
                # futures.append(executor.submit(self.evaluate_masks, batch, masks, selected_models_of_masks, model, problem, max_option_size, trajectories, number_actions, number_iterations))
                value, mask, length = self.evaluate_masks(batch, masks, selected_models_of_masks, model, problem, max_option_size, trajectories, number_actions, number_iterations)
                if best_mask is None or value < best_value:
                        best_value = value
                        best_mask = mask
                        best_length = length

        return best_mask, best_value, best_length


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

    logger.info(f'mask_transform_type="{args.mask_transform_type}, selection_type="{args.selection_type}"')

    module_extractor = DecOption(args, logger)
    module_extractor.discover()

    logger.info(f"Run id: {run_name}")
    logger.info(f"logs saved at {args.log_path}")


if __name__ == "__main__":
    main()