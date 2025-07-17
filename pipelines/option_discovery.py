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
    # exp_name: str = "extract_learnOption_noReg_excludeGoal"
    exp_name: str = "extract_learnOption_filtered"
    # exp_name: str = "debug"
    # exp_name: str = "extract_decOptionWhole_sparseInit"
    # exp_name: str = "extract_learnOptions_randomInit_discreteMasks"
    # exp_name: str = "extract_learnOptions_randomInit_pitisFunction"
    """the name of this experiment"""
    # env_seeds: Union[List, str, Tuple] = (0,1,2,3)
    # env_seeds: Union[List, str, Tuple] = (0,1,2)
    env_seeds: Union[List, str, Tuple] = (1,3,17)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
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
    # model_paths: List[str] = (
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    # model_paths: List[str] = (
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
    #     )
    
    # model_paths: List[str] = (
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    # )

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
    cpus: int = 1
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameters
    game_width: int = 9
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""

    # hill climbing arguments
    number_restarts: int = 400
    """number of hill climbing restarts for finding one option"""

    # mask learning
    mask_learning_rate: float = 0.0005
    """"""
    mask_learning_steps: int = 3_000
    """"""
    max_grad_norm: float = 1.0
    """"""
    input_update_frequency: int = 1
    """"""
    mask_type: str = "both"
    """It's one of these: [internal, input, both]"""
    mask_transform_type: str = "softmax"
    """It's either `softmax` or `quantize`"""
    selection_type: str = "local_search"
    """It's either `local_search` or `greedy`"""
    cache_path: str = ""
    """Path to the directory where the options are saved. If empty, it will be replaced based on the current `exp_id`"""
    # reg_coef: float = 0.0
    # reg_coef: float = 110.03 # Combogrid 4 environments
    reg_coef: float = 0.01
    filtering_inapplicable: bool = True
    # max_num_options: int = 10
    max_num_options: int = 5
    """"""
    option_mode:str = "didec-reg"

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}' + \
        f'_r{args.number_restarts}_envsd{",".join(map(str, args.env_seeds))}'
        if 'mask_type' in vars(args):
            args.exp_id += f'_mskType{args.mask_type}'
        if 'mask_transform_type' in vars(args):
            args.exp_id += f'_mskTransform{args.mask_transform_type}'
        if 'selection_type' in vars(args):
            args.exp_id += f'_selectType{args.selection_type}'
        args.exp_id += f'_reg{args.reg_coef}' # TODO: not conditioned correctly
        # args.exp_id += f'maxNumOptions{args.max_num_options}'

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
    print(args.env_seeds, args.problems, args.model_paths)
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{args.env_id}_width={args.game_width}_vanilla/seed={args.seed}/{model_directory}.pt'
        env = get_single_environment(args, seed=seed)
        print(env, seed)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path, weights_only=True))
        agent.eval()

        trajectory, _ = agent.run(env, verbose=verbose, length_cap=21)
        trajectories[problem] = trajectory

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")
        
        if len(trajectory.get_state_sequence()) > 20:
            logger.warning("The trajectory is longer than 20 steps, stopping the process..")
            exit()

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
    args = copy.deepcopy(args)
    if args.env_id == "MiniGrid-FourRooms-v0":
        args.env_id = "MiniGrid-SimpleCrossingS9N1-v0"
    elif args.env_id == "MiniGrid-MultiRoom-v0":
        args.env_id = "MiniGrid-Unlock-v0"
    save_dir = f"binary/options/{args.env_id}_width={args.game_width}_{args.option_mode}/seed={args.seed}"

    logger.info(f"Option directory: {save_dir}")

    model_files = sorted([f for f in os.listdir(save_dir) if f.startswith('ppo_model_option_') and f.endswith('.pth')])
    
    logger.info(f"Found options: {model_files}")

    n = len(model_files)
    options = [None] * n

    print(model_files)
    

    for model_file in model_files:
        model_path = os.path.join(save_dir, model_file)
        checkpoint = torch.load(model_path, weights_only=False)
        assert int(checkpoint['environment_args']['game_width']) == int(args.game_width)
        
        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0" or args.env_id == "MiniGrid-Unlock-v0":
            if 'environment_args' in checkpoint:
                seed = int(checkpoint['environment_args']['seed'])
            else:
                seed = int(checkpoint['problem'][-1])
            problem = None
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


def hill_climbing_iter(
    i: int, 
    agent: PPOAgent,
    option_size: int, 
    problem_str: str,
    number_actions: int, 
    mask_values: List, 
    trajectories: dict, 
    selected_masks: List, 
    selected_mask_models: List, 
    selected_option_sizes: List, 
    initial_loss: float, 
    args: Args, 
    loss: LevinLossActorCritic
):
    # Initialize the value depending on whether it's the last restart or not
    if i == args.number_restarts:
        mask_seq = [-1 for _ in range(args.hidden_size)]
    else:
        mask_seq = random.choices(mask_values, k=args.hidden_size)
    
    # Initialize the current mask
    current_mask = torch.tensor(mask_seq, dtype=torch.int8).view(1, -1)
    init_mask = current_mask
    applicable = False

    # Compute initial loss
    best_value = loss.compute_loss(
        selected_masks + [current_mask], 
        selected_mask_models + [agent], 
        problem_str, 
        trajectories, 
        number_actions, 
        number_steps=selected_option_sizes + [option_size]
    )

    # Check against default loss
    if best_value < initial_loss:
        applicable = True

    n_steps = 0
    while True:
        made_progress = False
        # Iterate through each element of the current mask
        for j in range(len(current_mask[0])):
            modifiable_current_mask = copy.deepcopy(current_mask)
            # Try each possible value for the current position
            for v in mask_values:
                if v == current_mask[0][j]:
                    continue
                
                modifiable_current_mask[0][j] = v
                eval_value = loss.compute_loss(
                    selected_masks + [modifiable_current_mask], 
                    selected_mask_models + [agent], 
                    problem_str, 
                    trajectories, 
                    number_actions, 
                    number_steps=selected_option_sizes + [option_size]
                )

                if eval_value < initial_loss:
                    applicable = True

                # Update the best value and mask if improvement is found
                if 'best_mask' not in locals() or eval_value < best_value:
                    best_value = eval_value
                    best_mask = copy.deepcopy(modifiable_current_mask)
                    made_progress = True

            # Update current mask to the best found so far
            current_mask = copy.deepcopy(best_mask)

        # Break the loop if no progress was made in the current iteration
        if not made_progress:
            break

        n_steps += 1

    # Optionally return the best mask and the best value if needed
    return i, best_value, current_mask, init_mask, n_steps, applicable
            

def hill_climbing(
        agent: PPOAgent, 
        problem_str: str,
        number_actions: int, 
        trajectories: dict, 
        selected_masks: List, 
        selected_masks_models: List, 
        selected_option_sizes: List, 
        possible_option_sizes: List, 
        loss: LevinLossActorCritic, 
        args: Args, 
        logger):
    """
    Performs Hill Climbing in the mask space for a given agent. Note that when computing the loss of a mask (option), 
    we pass the name of the problem in which the mask is used. That way, we do not evaluate an option on the problem in 
    which the option's model was trained. 

    Larger number of restarts will result in computationally more expensive searches, with the possibility of finding 
    a mask that better optimizes the Levin loss. 
    """

    best_mask = None
    mask_values = [-1, 0, 1]
    best_overall = None
    best_option_sizes = None
    best_value_overall = None
    
    def _update_best(i, best_value, best_mask, n_steps, init_mask, applicable, n_applicable, option_size):
        nonlocal best_overall, best_value_overall, best_option_sizes
        if applicable:
            n_applicable[i] = 1
        if i == args.number_restarts:
            logger.info(f'restart #{i}, Resulting Mask from the original Model: {best_mask}, Loss: {best_value}, using {option_size} iterations.')
        if best_overall is None or best_value < best_value_overall:
            best_overall = copy.deepcopy(best_mask)
            best_value_overall = best_value
            best_option_sizes = option_size

            logger.info(f'restart #{i}, Best Mask Overall: {best_overall}, Best Loss: {best_value_overall}, Best number of iterations: {best_option_sizes}')
            logger.info(f'restart #{i}, {n_steps} steps taken.\n Starting mask: {init_mask}\n Resulted Mask: {best_mask}')

    default_loss = loss.compute_loss(selected_masks, selected_masks_models, problem_str, trajectories, number_actions, number_steps=selected_option_sizes)
    
    for option_size in possible_option_sizes:
        logger.info(f'Selecting option #{len(selected_masks_models)} - option size {option_size}')
        n_applicable = [0] * (args.number_restarts + 1)

        if args.cpus == 1: 
            for i in range(args.number_restarts + 1):
                _, best_value, best_mask, init_mask, n_steps, applicable = hill_climbing_iter(i=i, 
                                                    agent=agent,
                                                    problem_str=problem_str,
                                                    option_size=option_size,
                                                    number_actions=number_actions,
                                                    mask_values=mask_values,
                                                    trajectories=trajectories,
                                                    selected_masks=selected_masks,
                                                    selected_mask_models=selected_masks_models,
                                                    selected_option_sizes=selected_option_sizes,
                                                    initial_loss=default_loss,
                                                    args=args,
                                                    loss=loss)
                _update_best(i=i, 
                             best_value=best_value, 
                             best_mask=best_mask,
                             n_steps=n_steps,
                             init_mask=init_mask,
                             applicable=applicable,
                             n_applicable=n_applicable,
                             option_size=option_size)
                
                if i % 100 == 0:
                    logger.info(f'Progress: {i}/{args.number_restarts}')
        else:
            # Use ProcessPoolExecutor to run the hill climbing iterations in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = [
                    executor.submit(
                        hill_climbing_iter, i, agent, option_size, problem_str, number_actions, 
                        mask_values, trajectories, selected_masks, selected_masks_models, 
                        selected_option_sizes, default_loss, args, loss
                    )
                    for i in range(args.number_restarts + 1)
                ]

                # Process the results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        i, best_value, best_mask, init_mask, n_steps, applicable = future.result()
                        _update_best(i=i, 
                                    best_value=best_value, 
                                    best_mask=best_mask,
                                    n_steps=n_steps, 
                                    init_mask=init_mask, 
                                    applicable=applicable,
                                    n_applicable=n_applicable, 
                                    option_size=option_size)
                        if i % 100 == 0:
                            logger.info(f'Progress: {i}/{args.number_restarts}')
                    except Exception as exc:
                        logger.error(f'restart #{i} generated an exception: {exc}')

        logger.info(f'Out of {args.number_restarts}, {sum(n_applicable)} options where applicable with size={option_size} .')
    
    return best_overall, best_value_overall, best_option_sizes


@timing_decorator
def hill_climbing_mask_space_training_data():
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    args = process_args()
    
    # Logger configurations
    logger = utils.get_logger('hill_climbing_logger', args.log_level, args.log_path)

    game_width = args.game_width
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])
    option_length = list(range(2, max_length + 1))
    # option_length = list(range(2, 5))
    args.exp_id += f'_olen{",".join(map(str, option_length))}'

    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    previous_loss = None
    best_loss = None

    # loss = LogitsLossActorCritic(logger)
    loss = LevinLossActorCritic(logger)

    selected_masks = []
    selected_mask_models = []
    selected_option_sizes = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
            model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
            logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
            env = get_single_environment(args, seed=seed)

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            mask, levin_loss, option_size = hill_climbing(agent=agent, 
                                                problem_str=problem, 
                                                number_actions=number_actions, 
                                                trajectories=trajectories, 
                                                selected_masks=selected_masks, 
                                                selected_masks_models=selected_mask_models, 
                                                selected_option_sizes=selected_option_sizes, 
                                                possible_option_sizes=option_length, 
                                                loss=loss, 
                                                args=args, 
                                                logger=logger)

            logger.info(f'Search Summary for {problem}, seed={seed}: \nBest Mask:{mask}, levin_loss={levin_loss}, n_iterations={option_size}\nPrevious Loss: {best_loss}, Previous selected loss:{previous_loss}, n_selected_masks={len(selected_masks)}')
            if best_loss is None or levin_loss < best_loss:
                best_loss = levin_loss
                best_mask_model = agent
                agent.to_option(mask, option_size, problem)
                if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
                    agent.environment_args = {
                        "seed": seed,
                        "game_width": game_width
                    }
                else:
                    agent.environment_args = {
                        "game_width": game_width
                    }
            utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_mask_models.append(best_mask_model)
        selected_option_sizes.append(best_mask_model.option_size)
        best_loss = loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args, 
                 logger=logger)
    
    utils.logger_flush(logger)


@timing_decorator
def hill_climbing_all_segments(args: Args, logger: logging.Logger):
    
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """

    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)

    logits_loss = LogitsLossActorCritic(logger)
    levin_loss = LevinLossActorCritic(logger)

    all_masks_info = []

    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
        env = get_single_environment(args, seed=seed)
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        agent.load_state_dict(torch.load(model_path))

        t_length = trajectories[problem].get_length()

        for length in range(2, t_length + 1):
            for s in range(t_length - length):
                logger.info(f"Processing option length={length}, segment={s}..")
                option_length = [length]
                sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
                mask, loss_value, option_size = hill_climbing(agent=agent, 
                                                problem_str="", 
                                                number_actions=number_actions, 
                                                trajectories=sub_trajectory, 
                                                selected_masks=[], 
                                                selected_masks_models=[], 
                                                selected_option_sizes=[], 
                                                possible_option_sizes=option_length, 
                                                loss=logits_loss, 
                                                args=args, 
                                                logger=logger)
                all_masks_info.append((mask, problem, option_size, model_path, s))
            utils.logger_flush(logger)
        
    logger.debug("\n")

    selected_masks = []
    selected_option_sizes = []
    selected_mask_models = []
    selected_options_problem = []

    previous_loss = None
    best_loss = None

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for mask, problem, option_size, model_path, segment in all_masks_info:
            logger.info(f'Extracting from the agent trained on problem={problem}, seed={seed}, segment=({segment},{segment+option_size})')
            env = get_single_environment(args, seed=seed)
            
            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            loss_value = levin_loss.compute_loss(masks=selected_masks + [mask], 
                                           agents=selected_mask_models + [agent], 
                                           problem_str=problem, 
                                           trajectories=trajectories, 
                                           number_actions=number_actions, 
                                           number_steps=selected_option_sizes + [option_size])


            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_mask_model = agent
                agent.to_option(mask, option_size, problem)

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_option_sizes.append(best_mask_model.option_size)
        selected_mask_models.append(best_mask_model)
        selected_options_problem.append(best_mask_model.problem_id)
        best_loss = levin_loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Added option #{len(selected_mask_models)}; Levin loss of the current selected set: {best_loss} on all trajectories")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args,  
                 logger=logger)
    
    utils.logger_flush(logger)

    levin_loss.print_output_subpolicy_trajectory(selected_mask_models, trajectories, logger=logger)
    utils.logger_flush(logger)


@timing_decorator
def whole_dec_options_training_data_levin_loss(args: Args, logger: logging.Logger):
    """
    This function performs hill climbing in the space of masks of a ReLU neural network
    to minimize the Levin loss of a given data set. 
    """
    number_actions = 3

    trajectories = regenerate_trajectories(args, verbose=True, logger=logger)
    max_length = max([len(t.get_trajectory()) for t in trajectories.values()])

    previous_loss = None
    best_loss = None

    loss = LevinLossActorCritic(logger)

    selected_masks = []
    selected_option_sizes = []
    selected_mask_models = []

    # the greedy loop of selecting options (masks)
    while previous_loss is None or best_loss < previous_loss:
        previous_loss = best_loss

        best_loss = None
        best_mask_model = None

        for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
            model_path = f'binary/models/{args.env_id}_width={args.game_width}_vanilla/seed={args.seed}/{model_directory}.pt'
            logger.info(f'Extracting from the agent trained on {problem}, seed={seed}')
            env = get_single_environment(args, seed=seed)

            agent = PPOAgent(env, hidden_size=args.hidden_size)
            agent.load_state_dict(torch.load(model_path))

            for i in range(2, max_length + 1):
                mask = torch.tensor([-1] * args.hidden_size).view(1,-1)
                levin_loss = loss.compute_loss(selected_masks + [mask], selected_mask_models + [agent], problem, trajectories, number_actions, selected_option_sizes + [i])
            
                if best_loss is None or levin_loss < best_loss:
                    best_loss = levin_loss
                    best_mask_model = agent
                    agent.to_option(mask, i, problem)

        logger.info(f'Summary of option #{len(selected_mask_models)}: \nBest Mask:{best_mask_model.mask}, best_loss={best_loss}, option_size={best_mask_model.option_size}, option problem={best_mask_model.problem_id}\nPrevious selected loss:{previous_loss}')
        utils.logger_flush(logger)
        logger.debug("\n")

        # we recompute the Levin loss after the automaton is selected so that we can use 
        # the loss on all trajectories as the stopping condition for selecting masks
        selected_masks.append(best_mask_model.mask)
        selected_mask_models.append(best_mask_model)
        selected_option_sizes.append(best_mask_model.option_size)
        best_loss = loss.compute_loss(selected_masks, selected_mask_models, "", trajectories, number_actions, selected_option_sizes)

        logger.info(f"Levin loss of the current set: {best_loss}")
        utils.logger_flush(logger)

    # remove the last automaton added
    num_options = len(selected_mask_models)
    selected_mask_models = selected_mask_models[:num_options - 1]

    # printing selected options
    logger.info("Selected options:")
    for i in range(len(selected_mask_models)):
        logger.info(f"Option #{i}:\n" + 
                    f"mask={selected_mask_models[i].mask}\n" +
                    f"size={selected_mask_models[i].option_size}\n" +
                    f"problem={selected_mask_models[i].problem_id}")

    save_options(options=selected_mask_models, 
                 trajectories=trajectories,
                 args=args, 
                 logger=logger)

    utils.logger_flush(logger)

    loss.print_output_subpolicy_trajectory(selected_mask_models, trajectories, logger=logger)
    utils.logger_flush(logger)


class STESoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Selects the highest probability (argmax) and returns a one-hot encoding.
        """
        max_indices = torch.argmax(x, dim=0, keepdim=True)
        
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(0, max_indices, 1.0)
        
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Allows gradient to flow through as if softmax was used.
        """
        return grad_output


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Define the quantization levels
        quantization_levels = torch.tensor([-1, 0, 1])
        distances = torch.abs(x.unsqueeze(1) - quantization_levels)
        quantized_indices = torch.argmin(distances, dim=1)
        return quantization_levels[quantized_indices]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LearnOptions:
    def __init__(self, args: Args, logger: logging.Logger):
        # args.option_mode = "didec"
        self.args = args
        self.logger = logger
        self.mask_transform_type = args.mask_transform_type
        self._mask_transform_func = self._get_transform_func()
        self.mask_type = args.mask_type
        self.levin_loss = LevinLossActorCritic(self.logger, alpha=args.reg_coef, mask_type=self.mask_type, mask_transform_type=self.mask_transform_type)
        self.number_actions = 3

        self.selection_type = args.selection_type
        if args.cache_path == "":
            args.cache_path = "binary/cache/"
            self.option_candidates_path = os.path.join(args.cache_path, f"{args.env_id}_width={args.game_width}_{args.option_mode}", f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"{args.env_id}_width={args.game_width}_{args.option_mode}", f"seed={args.seed}", "option_cache.pkl")
        else:
            self.option_candidates_path = os.path.join(args.cache_path, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"seed={args.seed}", "option_cache.pkl")

    def _get_transform_func(self):
        if self.mask_transform_type == "softmax":
            return LearnOptions._softmax_transform
        elif self.mask_transform_type == "quantize":
            return LearnOptions._tanh_transform
        else:
            raise ValueError(f"Invalid mask transform type: {self.mask_transform_type}")

    @staticmethod
    def _tanh_transform(mask):
        mask_transformed = 1.5 * torch.tanh(mask) + 0.5 * torch.tanh(-3 * mask)
        mask_transformed = STEQuantize.apply(mask_transformed)
        return mask_transformed

    @staticmethod
    def _softmax_transform(mask):
            return STESoftmax.apply(torch.softmax(mask, dim=0))

    @timing_decorator
    def discover(self):
        """
        This function performs hill climbing in the space of masks of a ReLU neural network
        to minimize the Levin loss of a given data set. It uses gumbel_softmax to extract 
        """

        if not os.path.exists(self.option_candidates_path):
            
            trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)

            option_candidates = []

            for target_seed, target_problem in zip(self.args.env_seeds, self.args.problems):
                self.logger.info(f'Extracting options using trajectory segments from {target_problem}, env_seed={target_seed}')
                mimicing_agents = {}

                t_length = trajectories[target_problem].get_length()

                for primary_seed, primary_problem, primary_model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
                    if primary_problem == target_problem:
                        continue
                    model_path = f'binary/models/{self.args.env_id}_width={self.args.game_width}_vanilla/seed={self.args.seed}/{primary_model_directory}.pt'
                    primary_env = get_single_environment(self.args, seed=primary_seed)
                    primary_agent = PPOAgent(primary_env, hidden_size=self.args.hidden_size)
                    primary_agent.load_state_dict(torch.load(model_path))
                    mimicing_agents[primary_problem] = (primary_seed, model_path, primary_agent)

                with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                    # Submit tasks to the executor with all required arguments
                    futures = []
                    for length in range(2, t_length + 1):
                        # if length != 3:
                        #     continue
                        for s in range(t_length - length + 1):
                            # actions = trajectories[target_problem].slice(s, n=length).get_action_sequence()
                            # if actions not in [[0,0,1], [2,1,0],[1,0,2], [0,1,2]]:
                            #     continue
                            for primary_problem, (primary_seed, primary_model_path, primary_agent) in mimicing_agents.items():
                                primary_agent.extra_info['primary_problem'] = primary_problem
                                future = executor.submit(
                                    self._train_mask_iter, trajectories, target_problem, s, length, primary_agent)
                                future.s = s,
                                future.length = length
                                future.primary_problem = primary_problem
                                future.primary_env_seed = primary_seed
                                future.primary_model_path = primary_model_path
                                futures.append(future)

                    # Process the results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            mask, init_loss, final_loss, applicable = future.result()
                            if (not self.args.filtering_inapplicable) or applicable:
                                option_candidates.append((mask, 
                                                    future.primary_problem, 
                                                    target_problem, 
                                                    future.primary_env_seed, 
                                                    target_seed, 
                                                    future.length, 
                                                    future.primary_model_path, 
                                                    future.s))
                            self.logger.info(f'Progress: segment:{future.s} of length {future.length}, primary_problem={future.primary_problem} done. init_loss={init_loss}, final_loss={final_loss}. applicable: {applicable}')
                        except Exception as exc:
                            self.logger.error(f'Segment:{future.s} of length {future.length} with primary_problem={future.primary_problem} generated an exception: {exc}')
                            traceback.print_exc()
                            return
                utils.logger_flush(self.logger)
            self.logger.debug("\n")
            self.logger.info("Saving parameters ... ")
            os.makedirs(os.path.dirname(self.option_candidates_path), exist_ok=True)
            with open(self.option_candidates_path, 'wb') as f:
                pickle.dump({'option_candidates': option_candidates, 'trajectories': trajectories}, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.info("Loading parameters ... ")
            with open(self.option_candidates_path, 'rb') as f:
                data = pickle.load(f)
                option_candidates = data['option_candidates']
                trajectories = data['trajectories']
                trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)
            if self.args.filtering_inapplicable and self.args.env_id == "ComboGrid":
                assert len(option_candidates) < 3300, f"Filtering inapplicable options is not working, {len(option_candidates)} options found. Please check the filtering logic."
            

        ### SANITY CHECK:
        # for i, option in enumerate(option_candidates):
        #     mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment = option
        #     segment = segment[0]
        #     env = get_single_environment(self.args, seed=primary_env_seed)
        #     agent = PPOAgent(env, hidden_size=self.args.hidden_size)
        #     agent.load_state_dict(torch.load(model_path))
        #     agent.to_option(mask, option_size, target_problem)
        #     if self.mask_type == "input":
        #         trajectory = agent.run_with_input_mask_softmax(trajectories[target_problem].slice(segment, n=option_size).get_state_sequence(),
        #                                       mask=mask,
        #                                       max_size_sequence=option_size)
        #     elif self.mask_type == "internal":
        #         trajectory = agent.run_with_mask_softmax(trajectories[target_problem].slice(segment, n=option_size).get_state_sequence(),
        #                                       mask=mask,
        #                                       max_size_sequence=option_size)
        #     elif self.mask_type == "both":
        #         trajectory = agent.run_with_both_masks_softmax(trajectories[target_problem].slice(segment, n=option_size).get_state_sequence(),
        #                                       input_mask=mask[0],
        #                                       internal_mask=mask[1],
        #                                       max_size_sequence=option_size)
        #     if trajectory.get_action_sequence() != trajectories[target_problem].slice(segment, n=option_size).get_action_sequence():
        #         self.logger.error(f"Mask does not match the original trajectory for target problem {target_problem} with segment {segment} and length {option_size}; primary_problem:{primary_problem}, expected: {trajectory.get_action_sequence()}, received: {trajectories[target_problem].slice(segment, n=option_size).get_action_sequence()}")
        #     # assert trajectory.get_action_sequence() == trajectories[target_problem].slice(segment, n=option_size).get_action_sequence(), \
        #     #     f"Mask {mask} does not match the original trajectory for problem {target_problem} with segment {segment} and length {option_size}, {trajectory.get_action_sequence()}, {trajectories[target_problem].slice(segment, n=option_size).get_action_sequence()}"
        # return
        ###

        if self.selection_type == "greedy":
            selected_options = self.select_greedy(option_candidates, trajectories)
        elif self.selection_type == "local_search":
            selected_options = self.select_by_local_search(option_candidates, trajectories)
        else:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        
        assert len(selected_options) > 0

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

        # self.logger.info("Testing on each grid cell")
        # for seed, problem in zip(self.args.env_seeds, self.args.problems):
        #     self.logger.info(f"Testing on each cell..., {problem}")
        #     self.levin_loss.evaluate_on_each_cell(options=selected_options, 
        #                             trajectories=trajectories,
        #                             problem_test=problem, 
        #                             args=self.args, 
        #                             seed=seed, 
        #                             logger=self.logger)

        utils.logger_flush(self.logger)

    def select_greedy(self, option_candidates, trajectories):
        selected_masks = []
        selected_option_sizes = []
        selected_options = []
        selected_options_problem = []

        previous_loss = None
        best_loss = None

        # the greedy loop of selecting options (masks)
        while previous_loss is None or best_loss < previous_loss:
            previous_loss = best_loss

            best_loss = None
            best_mask_model = None

            for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
                self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
                env = get_single_environment(self.args, seed=primary_env_seed)
                agent = PPOAgent(env, hidden_size=self.args.hidden_size)
                agent.load_state_dict(torch.load(model_path))

                loss_value = self.levin_loss.compute_loss(masks=selected_masks + [mask], 
                                            agents=selected_options + [agent], 
                                            problem_str="", 
                                            trajectories=trajectories, 
                                            number_actions=self.number_actions, 
                                            number_steps=selected_option_sizes + [option_size])

                if best_loss is None or loss_value < best_loss:
                    best_loss = loss_value
                    best_mask_model = agent
                    agent.to_option(mask, option_size, target_problem)
                    agent.extra_info['primary_problem'] = primary_problem
                    agent.extra_info['primary_env_seed'] = primary_env_seed
                    agent.extra_info['target_problem'] = target_problem
                    agent.extra_info['target_env_seed'] = target_env_seed
                    agent.extra_info['segment'] = segment

            # we recompute the Levin loss after the automaton is selected so that we can use 
            # the loss on all trajectories as the stopping condition for selecting masks
            selected_masks.append(best_mask_model.mask)
            selected_option_sizes.append(best_mask_model.option_size)
            selected_options.append(best_mask_model)
            selected_options_problem.append(best_mask_model.problem_id)
            best_loss = self.levin_loss.compute_loss(selected_masks, selected_options, "", trajectories, self.number_actions, selected_option_sizes)

            if previous_loss is None or best_loss < previous_loss:
                self.logger.info(f"Added option #{len(selected_options)}; Levin loss of the current selected set: {best_loss} on all trajectories")
            utils.logger_flush(self.logger)

        # remove the last automaton added
        num_options = len(selected_options)
        selected_options = selected_options[:num_options - 1]
        return selected_options

    def _compute_sample_weight(self, option_refs, possible_sequences, all_options):
        transitions = [0.1 for _ in range(len(option_refs))]
        for i, o_idx in enumerate(option_refs):
            option = all_options[o_idx]
            option_id = option.get_option_id()
            for problem_name, sub_cache in self.levin_loss.cache[option_id].items():
                for segment, (applicable, _) in sub_cache.items():
                    if applicable == True and (segment, segment + option.option_size) in possible_sequences[problem_name]:
                        transitions[i] += 1

        weights = np.array(transitions)
        weights /= weights.sum()
        return weights

    def _search_options_subset(self, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, worker_id):
        max_num_neighbours = 125
        
        random_generator = np.random.default_rng([worker_id, self.args.seed])
        all_option_refs = set(range(len(all_options)))
        
        max_num_options = min(max_num_options, len(all_option_refs))
        # length_weights = np.array([1/(i+2) for i in range(max_num_options + 1)])
        length_weights = np.array([1/(i*3+2) for i in range(max_num_options + 1)])
        length_weights /= np.sum(length_weights)
        subset_length = random_generator.choice(range(max_num_options + 1), p=length_weights)
        
        weights = self._compute_sample_weight(all_option_refs, all_possible_sequences, all_options)
        selected_options = set(random_generator.choice(list(all_option_refs), p=weights, size=subset_length, replace=False).tolist())
        possible_sequences = copy.deepcopy(all_possible_sequences)
        
        previous_cost = float('Inf')
        total_loss_calculations = 1
        steps = 0
        best_cost = 0
        for problem, trajectory in trajectories.items():
            cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in selected_options], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
            possible_sequences[problem] = possible_sequences[problem] - used_sequences
            best_cost += cost
        
        while (best_cost < previous_cost or steps == 0) and steps < max_steps:
            previous_cost = best_cost
            not_selected_options = all_option_refs - selected_options
            weights = self._compute_sample_weight(not_selected_options, possible_sequences, all_options)
            num_neighbours = min(max_num_neighbours, len(not_selected_options))
            sample_options = random_generator.choice(list(not_selected_options), p=weights, size=num_neighbours, replace=False).tolist()
            # assert len(sample_options) == (len(all_options) - len(selected_options)) == len(not_selected_options), f"Sampled options {len(sample_options)} should be equal to all options {len(all_options)} len of selected ones: {len(selected_options)}, len of not selected ones: {len(not_selected_options)}"

            neighbours = []
            for option in sample_options:
                assert option not in selected_options, f"Option {option.get_option_id()} should not be in selected options."
                if option not in selected_options:
                    if len(selected_options) < max_num_options:
                        neighbour = selected_options | {option}
                        neighbours.append(neighbour)
                    for option2 in selected_options:
                        neighbour = selected_options - {option2} | {option}
                        neighbours.append(neighbour)
            for option in selected_options:
                neighbour = selected_options - {option}
                neighbours.append(neighbour)
                
            # self.logger.info(f"Number of neighbours: {len(neighbours)}")
            i = 0
            for neighbour in neighbours:
                i += 1
                cost = 0
                remaining_sequences = copy.deepcopy(all_possible_sequences)
                for problem, trajectory in trajectories.items():
                    returned_cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in neighbour], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
                    remaining_sequences[problem] = remaining_sequences[problem] - used_sequences
                    cost += returned_cost
                    total_loss_calculations += 1
                if cost < best_cost:
                    selected_options = neighbour
                    possible_sequences = remaining_sequences
                    best_cost = cost
            steps += 1
        
        return best_cost, selected_options, {"total_loss_calculations": total_loss_calculations, "steps":steps}

    def _compute_option_applicability(self, option, trajectories):
        result = {problem: {} for problem in trajectories.keys()}
        applicable_count = 0
        for problem, trajectory in trajectories.items():
            t_len = trajectory.get_length()
            t = trajectory.get_trajectory()
            for s in range(t_len):
                actions = self.levin_loss._run(copy.deepcopy(t[s][0]), option.mask, option, option.option_size)
                is_applicable = (len(actions) == option.option_size) and self.levin_loss.is_applicable(t, actions, s)
                if is_applicable:
                    applicable_count += 1
                result[problem][s] = (is_applicable, actions)
        return result, applicable_count

    def select_by_local_search(self, option_candidates, trajectories):
        all_options = []
        total_applicable_count = 0
        for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
            self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
            env = get_single_environment(self.args, seed=primary_env_seed)
            agent = PPOAgent(env, hidden_size=self.args.hidden_size)
            agent.load_state_dict(torch.load(model_path))
            agent.to_option(mask, option_size, target_problem)
            agent.extra_info['primary_problem'] = primary_problem
            agent.extra_info['primary_env_seed'] = primary_env_seed
            agent.extra_info['target_problem'] = target_problem
            agent.extra_info['target_env_seed'] = target_env_seed
            agent.extra_info['segment'] = segment
            all_options.append(agent)
            option = agent
            option_id = option.get_option_id()
            self.levin_loss.cache[option_id], n_applicable = self._compute_option_applicability(option, trajectories)
            total_applicable_count += n_applicable

        self.logger.info(f"Number of option_candidates: {len(all_options)}")

        if os.path.exists(self.option_cache_path):
            self.logger.info(f"Loading option cache from {self.option_cache_path}")
            with open(self.option_cache_path, 'rb') as f:
                self.levin_loss.cache = pickle.load(f)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = set()
                for i, option in enumerate(all_options):
                    future = executor.submit(
                        self._compute_option_applicability, option, trajectories)
                    future.option_id = option.get_option_id()
                    futures.add(future)
                self.logger.info(f"Logging Checkpoint 1.")

                # Process the results as they complete
                total_applicable_count = 0
                progress = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        option_id = future.option_id
                        self.levin_loss.cache[option_id], n_applicable = future.result()
                        total_applicable_count += n_applicable
                        progress += 1
                        if progress % 100 == 0:
                            self.logger.info(f"Cache size: {len(self.levin_loss.cache)}, applicable_count: {total_applicable_count}")
                    except Exception as exc:
                        self.logger.error(f'Exception: {exc}')
                        traceback.print_exc()
                        return

            # applicable_count = 0
            # for i, option in enumerate(all_options):
            #     option_id = option.get_option_id()
            #     for problem, trajectory in trajectories.items():
            #         t_len = trajectory.get_length()
            #         t = trajectory.get_trajectory()
            #         for s in range(t_len):
            #             if option_id not in self.levin_loss.cache:
            #                 self.levin_loss.cache[option_id] = {}
            #             if problem not in self.levin_loss.cache[option_id]:
            #                 self.levin_loss.cache[option_id][problem] = {}
            #             actions = self.levin_loss._run(copy.deepcopy(t[s][0]), option.mask, option, option.option_size)
            #             is_applicable = (len(actions) == option.option_size) and self.levin_loss.is_applicable(t, actions, s)
            #             if is_applicable:
            #                 applicable_count += 1
            #             self.levin_loss.cache[option.get_option_id()][problem][s] = (is_applicable, actions)
            #     if i % 100 == 0:
            #         self.logger.info(f"Cache size: {len(self.levin_loss.cache)}, applicable_count: {applicable_count}")
            
            # self.logger.info(f"Saving option cache to {self.option_cache_path}")
            # os.makedirs(os.path.dirname(self.option_cache_path), exist_ok=True)
            # with open(self.option_cache_path, 'wb') as f:
            #     pickle.dump(self.levin_loss.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Cache created. size: {len(self.levin_loss.cache)}")

        all_possible_sequences = {}
        for problem_name, trajectory in trajectories.items():
            trajectory = trajectory.get_trajectory()
            all_possible_sequences[problem_name] = set()
            t_length = len(trajectory)
            for length in range(2, t_length + 1):
                for s in range(0, t_length - length + 1):
                    all_possible_sequences[problem_name].add((s, s+length))

        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list

        restarts = 200
        max_steps = 500
        max_num_options = self.args.max_num_options
        best_selected_options = []
        best_levin_loss_total = float('Inf')
        completed = 0

        for problem, trajectory in trajectories.items():
            cost, _ = self.levin_loss.compute_loss_cached([], 
                                                            trajectory, 
                                                            problem_str=problem, 
                                                            number_actions=self.number_actions)
            print(f"Cost of empty set: {cost} for problem {problem}")
            

        # for i in range(10):
        #     best_cost, selected_options, info = self._search_options_subset(max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
        #     total_loss_calculations = info['total_loss_calculations']
        #     steps = info['steps']
        #     # self.levin_loss.cache.update(cache)
        #     if best_cost < best_levin_loss_total:
        #         best_levin_loss_total = best_cost
        #         best_selected_options = selected_options
        #     completed += 1
        #     # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
        #     self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
        #     utils.logger_flush(self.logger)        


        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
            # Submit tasks to the executor with all required arguments
            futures = set()
            for i in range(restarts):
                future = executor.submit(
                    self._search_options_subset, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
                self.logger.info(f"Restart {i} of {restarts} submitted.")
                futures.add(future)
            self.logger.info(f"Logging Checkpoint 2.")

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    best_cost, selected_options, info = future.result()
                    total_loss_calculations = info['total_loss_calculations']
                    steps = info['steps']
                    # self.levin_loss.cache.update(cache)
                    if best_cost < best_levin_loss_total:
                        best_levin_loss_total = best_cost
                        best_selected_options = selected_options
                    completed += 1
                    # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
                    self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
                    utils.logger_flush(self.logger)
                except Exception as exc:
                    self.logger.error(f'Exception: {exc}')
                    traceback.print_exc()
                    return

        best_selected_options = [all_options[idx] for idx in best_selected_options]
        self.levin_loss.remove_cache()

        # Removing redundant options
        def get_levin_loss(options, trajectories):
            cost = 0
            for problem, trajectory in trajectories.items():
                cost += self.levin_loss.compute_loss_cached(options, 
                                                trajectory, 
                                                problem_str=problem, 
                                                number_actions=3,
                                                cache_enabled=False)[0]
            return cost
    
        best_levin_loss = get_levin_loss(best_selected_options, trajectories)
        
        self.logger.info(f"Levin loss: {best_levin_loss}")
        options = copy.deepcopy(best_selected_options)
        while True:
            done = True
            best_loss_so_far = best_levin_loss
            for i in range(len(options)):
                options_cpy = copy.deepcopy(options)
                options_cpy = options_cpy[:i] + options_cpy[i+1:]
                levin_loss = get_levin_loss(options_cpy, trajectories)
                if levin_loss < best_loss_so_far:
                    best_loss_so_far = levin_loss
                    best_options_so_far = options_cpy
                    redundant_idx = i
                    done = False
            if not done:
                best_levin_loss = best_loss_so_far
                options = best_options_so_far
                self.logger.info(f"Levin loss without option #{redundant_idx}: {best_levin_loss}")
            else:
                break
                    


        return list(best_selected_options)

    def _train_mask_iter(self, trajectories, problem, s, length, agent: PPOAgent):
        sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
        # learn option with this length
        if self.mask_type != "both":
            if self.mask_type == "internal" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask
            elif self.mask_type == "internal" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask_softmax
            elif self.mask_type == "input" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask
            elif self.mask_type == "input" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask_softmax
            return self._train_mask(mask, agent, sub_trajectory, rollout_func)
        else:
            if self.mask_transform_type == "quantize":
                input_mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks
            elif self.mask_transform_type == "softmax":
                input_mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks_softmax
            return self._train_mask_both(input_mask, internal_mask, agent, sub_trajectory, rollout_func)

    def _train_mask(self, mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        # mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
        optimizer = torch.optim.Adam([mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                mask_transformed = self._mask_transform_func(mask)

                # Generate a rollout with the mask
                new_trajectory = rollout_func(envs, mask_transformed, trajectory.get_length())

                # Calculate the mask loss            
                # input_probs = torch.nn.functional.log_softmax(torch.stack(new_trajectory.logits), dim=0)
                # target_probs = torch.nn.functional.softmax(torch.stack(trajectory.logits) , dim=0)
                # loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                # mask_loss = loss_fn(input_probs, target_probs)
                loss_fn = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions)) + self.args.reg_coef * mask[-1, :].sum()**2
                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_mask = mask_transformed

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_mask = mask_transformed
                optimizer.zero_grad()
                mask_loss.backward()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)
                optimizer.step()

                # Update progress
                steps += 1
            # self.logger.info(f"Steps: {steps}, Best loss: {best_loss}")

        # assert trajectory.get_action_sequence() == new_trajectory.get_action_sequence(), \
        #     f"Mask {mask} does not match the original trajectory for target problem {trajectories.keys()} , \
        #         {new_trajectory.get_action_sequence()}, expected: {trajectory.get_action_sequence()} \n \
        #             primary_problem={agent.extra_info['primary_problem']}"
        trajectory = list(trajectories.values())[0]
        envs = trajectory.get_state_sequence()
        new_trajectory = rollout_func(envs, best_mask, trajectory.get_length())

        applicable = new_trajectory.get_action_sequence() == trajectory.get_action_sequence()
        return best_mask.detach().data, init_loss, best_loss, applicable

    def _train_mask_both(self, input_mask, internal_mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        input_optimizer = torch.optim.Adam([input_mask], lr=self.args.mask_learning_rate)
        internal_optimizer = torch.optim.Adam([internal_mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_input_mask = None
        best_internal_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                input_mask_discretized = self._mask_transform_func(input_mask)

                internal_mask_discretized = self._mask_transform_func(internal_mask)

                new_trajectory = rollout_func(envs, input_mask_discretized, internal_mask_discretized, trajectory.get_length())

                loss_fn = torch.nn.CrossEntropyLoss()
                # mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions)) 
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions)) + self.args.reg_coef * input_mask[-1, :].sum()**2

                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                internal_optimizer.zero_grad()
                mask_loss.backward(retain_graph=True)  
                internal_optimizer.step()

                if steps % self.args.input_update_frequency == 0:  
                    input_optimizer.zero_grad()
                    mask_loss.backward()  
                    input_optimizer.step()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)

                # Update progress
                steps += 1
        
        trajectory = list(trajectories.values())[0]
        envs = trajectory.get_state_sequence()
        new_trajectory = rollout_func(envs, best_input_mask, best_internal_mask, trajectory.get_length())

        applicable = new_trajectory.get_action_sequence() == trajectory.get_action_sequence()
        return (best_input_mask.detach().data, best_internal_mask.detach().data), init_loss, mask_loss.item(), applicable



class WholeDecOption:
    def __init__(self, args: Args, logger: logging.Logger):
        args.option_mode = "dec-whole"
        self.args = args
        self.logger = logger
        self.mask_transform_type = args.mask_transform_type
        self._mask_transform_func = self._get_transform_func()
        self.mask_type = args.mask_type
        self.levin_loss = LevinLossActorCritic(self.logger, alpha=args.reg_coef, mask_type=self.mask_type, mask_transform_type=self.mask_transform_type)
        self.number_actions = 3

        self.selection_type = args.selection_type
        if args.cache_path == "":
            args.cache_path = "binary/cache/"
            self.option_candidates_path = os.path.join(args.cache_path, f"{args.env_id}_width={args.game_width}_{args.option_mode}", f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"{args.env_id}_width={args.game_width}_{args.option_mode}", f"seed={args.seed}", "option_cache.pkl")
        else:
            self.option_candidates_path = os.path.join(args.cache_path, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"seed={args.seed}", "option_cache.pkl")

    def _get_transform_func(self):
        if self.mask_transform_type == "softmax":
            return LearnOptions._softmax_transform
        elif self.mask_transform_type == "quantize":
            return LearnOptions._tanh_transform
        else:
            raise ValueError(f"Invalid mask transform type: {self.mask_transform_type}")

    @staticmethod
    def _tanh_transform(mask):
        mask_transformed = 1.5 * torch.tanh(mask) + 0.5 * torch.tanh(-3 * mask)
        mask_transformed = STEQuantize.apply(mask_transformed)
        return mask_transformed

    @staticmethod
    def _softmax_transform(mask):
            return STESoftmax.apply(torch.softmax(mask, dim=0))

    @timing_decorator
    def discover(self):
        """
        This function performs hill climbing in the space of masks of a ReLU neural network
        to minimize the Levin loss of a given data set. It uses gumbel_softmax to extract 
        """

        trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)
        option_candidates = []
        for primary_env_seed, primary_problem, primary_model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
            t_length = trajectories[primary_problem].get_length()
            model_path = f'binary/models/{self.args.env_id}_width={self.args.game_width}_vanilla/seed={self.args.seed}/{primary_model_directory}.pt'
            if self.mask_transform_type == "quantize":
                mask = torch.zeros(self.args.hidden_size) - 1
            elif self.mask_transform_type == "softmax":
                mask = torch.zeros(3, self.args.hidden_size)
                mask[-1] = 1
            
            for length in range(2, t_length + 1):
                option_candidates.append((mask, 
                    primary_problem, 
                    "", 
                    primary_env_seed, 
                    -1, 
                    length, 
                    model_path, 
                    (-1,)))

        if self.selection_type == "greedy":
            selected_options = self.select_greedy(option_candidates, trajectories)
        elif self.selection_type == "local_search":
            selected_options = self.select_by_local_search(option_candidates, trajectories)
        else:
            raise ValueError(f"Invalid selection type: {self.selection_type}")
        
        assert len(selected_options) > 0

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

        # self.logger.info("Testing on each grid cell")
        # for seed, problem in zip(self.args.env_seeds, self.args.problems):
        #     self.logger.info(f"Testing on each cell..., {problem}")
        #     self.levin_loss.evaluate_on_each_cell(options=selected_options, 
        #                             trajectories=trajectories,
        #                             problem_test=problem, 
        #                             args=self.args, 
        #                             seed=seed, 
        #                             logger=self.logger)

        utils.logger_flush(self.logger)

    def select_greedy(self, option_candidates, trajectories):
        selected_masks = []
        selected_option_sizes = []
        selected_options = []
        selected_options_problem = []

        previous_loss = None
        best_loss = None

        # the greedy loop of selecting options (masks)
        while previous_loss is None or best_loss < previous_loss:
            previous_loss = best_loss

            best_loss = None
            best_mask_model = None

            for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
                self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
                env = get_single_environment(self.args, seed=primary_env_seed)
                agent = PPOAgent(env, hidden_size=self.args.hidden_size)
                agent.load_state_dict(torch.load(model_path))

                loss_value = self.levin_loss.compute_loss(masks=selected_masks + [mask], 
                                            agents=selected_options + [agent], 
                                            problem_str="", 
                                            trajectories=trajectories, 
                                            number_actions=self.number_actions, 
                                            number_steps=selected_option_sizes + [option_size])

                if best_loss is None or loss_value < best_loss:
                    best_loss = loss_value
                    best_mask_model = agent
                    agent.to_option(mask, option_size, target_problem)
                    agent.extra_info['primary_problem'] = primary_problem
                    agent.extra_info['primary_env_seed'] = primary_env_seed
                    agent.extra_info['target_problem'] = target_problem
                    agent.extra_info['target_env_seed'] = target_env_seed
                    agent.extra_info['segment'] = segment

            # we recompute the Levin loss after the automaton is selected so that we can use 
            # the loss on all trajectories as the stopping condition for selecting masks
            selected_masks.append(best_mask_model.mask)
            selected_option_sizes.append(best_mask_model.option_size)
            selected_options.append(best_mask_model)
            selected_options_problem.append(best_mask_model.problem_id)
            best_loss = self.levin_loss.compute_loss(selected_masks, selected_options, "", trajectories, self.number_actions, selected_option_sizes)

            if previous_loss is None or best_loss < previous_loss:
                self.logger.info(f"Added option #{len(selected_options)}; Levin loss of the current selected set: {best_loss} on all trajectories")
            utils.logger_flush(self.logger)

        # remove the last automaton added
        num_options = len(selected_options)
        selected_options = selected_options[:num_options - 1]
        return selected_options

    def _compute_sample_weight(self, option_refs, possible_sequences, all_options):
        transitions = [0.1 for _ in range(len(option_refs))]
        for i, o_idx in enumerate(option_refs):
            option = all_options[o_idx]
            option_id = option.get_option_id()
            for problem_name, sub_cache in self.levin_loss.cache[option_id].items():
                for segment, (applicable, _) in sub_cache.items():
                    if applicable == True and (segment, segment + option.option_size) in possible_sequences[problem_name]:
                        transitions[i] += 1

        weights = np.array(transitions)
        weights /= weights.sum()
        return weights

    def _search_options_subset(self, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, worker_id):
        max_num_neighbours = 1000
        
        random_generator = np.random.default_rng([worker_id, self.args.seed])
        all_option_refs = set(range(len(all_options)))
        
        max_num_options = min(max_num_options, len(all_option_refs))
        # length_weights = np.array([1/(i+2) for i in range(max_num_options + 1)])
        length_weights = np.array([1/(i*3+2) for i in range(max_num_options)])
        length_weights /= np.sum(length_weights)
        subset_length = random_generator.choice(range(1, max_num_options + 1), p=length_weights)
        
        weights = self._compute_sample_weight(all_option_refs, all_possible_sequences, all_options)
        selected_options = set(random_generator.choice(list(all_option_refs), p=weights, size=subset_length, replace=False).tolist())
        possible_sequences = copy.deepcopy(all_possible_sequences)
        
        previous_cost = float('Inf')
        total_loss_calculations = 1
        steps = 0
        best_cost = 0
        for problem, trajectory in trajectories.items():
            cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in selected_options], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
            possible_sequences[problem] = possible_sequences[problem] - used_sequences
            best_cost += cost
        
        while (best_cost < previous_cost or steps == 0) and steps < max_steps:
            previous_cost = best_cost
            not_selected_options = all_option_refs - selected_options
            weights = self._compute_sample_weight(not_selected_options, possible_sequences, all_options)
            num_neighbours = min(max_num_neighbours, len(not_selected_options))
            sample_options = random_generator.choice(list(not_selected_options), p=weights, size=num_neighbours, replace=False).tolist()
            # assert len(sample_options) == (len(all_options) - len(selected_options)) == len(not_selected_options), f"Sampled options {len(sample_options)} should be equal to all options {len(all_options)} len of selected ones: {len(selected_options)}, len of not selected ones: {len(not_selected_options)}"

            neighbours = []
            for option in sample_options:
                assert option not in selected_options, f"Option {option.get_option_id()} should not be in selected options."
                if option not in selected_options:
                    if len(selected_options) < max_num_options:
                        neighbour = selected_options | {option}
                        neighbours.append(neighbour)
                    for option2 in selected_options:
                        neighbour = (selected_options - {option2}) | {option}
                        neighbours.append(neighbour)
            if len(selected_options) > 1:
                for option in selected_options:
                    neighbour = selected_options - {option}
                    neighbours.append(neighbour)
                
            # self.logger.info(f"Number of neighbours: {len(neighbours)}")
            i = 0
            for neighbour in neighbours:
                i += 1
                cost = 0
                remaining_sequences = copy.deepcopy(all_possible_sequences)
                for problem, trajectory in trajectories.items():
                    returned_cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in neighbour], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
                    remaining_sequences[problem] = remaining_sequences[problem] - used_sequences
                    cost += returned_cost
                    total_loss_calculations += 1
                if cost < best_cost:
                    selected_options = neighbour
                    possible_sequences = remaining_sequences
                    best_cost = cost
            steps += 1
        
        return best_cost, selected_options, {"total_loss_calculations": total_loss_calculations, "steps":steps}

    def _compute_option_applicability(self, option, trajectories):
        result = {problem: {} for problem in trajectories.keys()}
        applicable_count = 0
        for problem, trajectory in trajectories.items():
            t_len = trajectory.get_length()
            t = trajectory.get_trajectory()
            for s in range(t_len):
                actions = self.levin_loss._run(copy.deepcopy(t[s][0]), option.mask, option, option.option_size)
                is_applicable = (len(actions) == option.option_size) and self.levin_loss.is_applicable(t, actions, s)
                if is_applicable:
                    applicable_count += 1
                result[problem][s] = (is_applicable, actions)
        return result, applicable_count

    def select_by_local_search(self, option_candidates, trajectories):
        all_options = []

        for mask, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
            self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
            env = get_single_environment(self.args, seed=primary_env_seed)
            agent = PPOAgent(env, hidden_size=self.args.hidden_size)
            agent.load_state_dict(torch.load(model_path))
            agent.to_option(mask, option_size, target_problem)
            agent.extra_info['primary_problem'] = primary_problem
            agent.extra_info['primary_env_seed'] = primary_env_seed
            agent.extra_info['target_problem'] = target_problem
            agent.extra_info['target_env_seed'] = target_env_seed
            agent.extra_info['segment'] = segment
            all_options.append(agent)

        self.logger.info(f"Number of option_candidates: {len(all_options)}")

        if os.path.exists(self.option_cache_path):
            self.logger.info(f"Loading option cache from {self.option_cache_path}")
            with open(self.option_cache_path, 'rb') as f:
                self.levin_loss.cache = pickle.load(f)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = set()
                for i, option in enumerate(all_options):
                    future = executor.submit(
                        self._compute_option_applicability, option, trajectories)
                    future.option_id = option.get_option_id()
                    futures.add(future)
                self.logger.info(f"Logging Checkpoint 1.")

                # Process the results as they complete
                total_applicable_count = 0
                progress = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        option_id = future.option_id
                        self.levin_loss.cache[option_id], n_applicable = future.result()
                        total_applicable_count += n_applicable
                        progress += 1
                        if progress % 100 == 0:
                            self.logger.info(f"Cache size: {len(self.levin_loss.cache)}, applicable_count: {total_applicable_count}")
                    except Exception as exc:
                        self.logger.error(f'Exception: {exc}')
                        traceback.print_exc()
                        return

            # applicable_count = 0
            # for i, option in enumerate(all_options):
            #     option_id = option.get_option_id()
            #     for problem, trajectory in trajectories.items():
            #         t_len = trajectory.get_length()
            #         t = trajectory.get_trajectory()
            #         for s in range(t_len):
            #             if option_id not in self.levin_loss.cache:
            #                 self.levin_loss.cache[option_id] = {}
            #             if problem not in self.levin_loss.cache[option_id]:
            #                 self.levin_loss.cache[option_id][problem] = {}
            #             actions = self.levin_loss._run(copy.deepcopy(t[s][0]), option.mask, option, option.option_size)
            #             is_applicable = (len(actions) == option.option_size) and self.levin_loss.is_applicable(t, actions, s)
            #             if is_applicable:
            #                 applicable_count += 1
            #             self.levin_loss.cache[option.get_option_id()][problem][s] = (is_applicable, actions)
            #     if i % 100 == 0:
            #         self.logger.info(f"Cache size: {len(self.levin_loss.cache)}, applicable_count: {applicable_count}")
            self.logger.info(f"Saving option cache to {self.option_cache_path}")
            os.makedirs(os.path.dirname(self.option_cache_path), exist_ok=True)
            with open(self.option_cache_path, 'wb') as f:
                pickle.dump(self.levin_loss.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Cache created. size: {len(self.levin_loss.cache)}")

        all_possible_sequences = {}
        for problem_name, trajectory in trajectories.items():
            trajectory = trajectory.get_trajectory()
            all_possible_sequences[problem_name] = set()
            t_length = len(trajectory)
            for length in range(2, t_length + 1):
                for s in range(0, t_length - length + 1):
                    all_possible_sequences[problem_name].add((s, s+length))

        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list

        restarts = 200
        max_steps = 500
        max_num_options = self.args.max_num_options
        best_selected_options = []
        best_levin_loss_total = float('Inf')
        completed = 0

        for problem, trajectory in trajectories.items():
            cost, _ = self.levin_loss.compute_loss_cached([], 
                                                            trajectory, 
                                                            problem_str=problem, 
                                                            number_actions=self.number_actions)
            print(f"Cost of empty set: {cost} for problem {problem}")
            

        # for i in range(10):
        #     best_cost, selected_options, info = self._search_options_subset(max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
        #     total_loss_calculations = info['total_loss_calculations']
        #     steps = info['steps']
        #     # self.levin_loss.cache.update(cache)
        #     if best_cost < best_levin_loss_total:
        #         best_levin_loss_total = best_cost
        #         best_selected_options = selected_options
        #     completed += 1
        #     # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
        #     self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
        #     utils.logger_flush(self.logger)        


        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
            # Submit tasks to the executor with all required arguments
            futures = set()
            for i in range(restarts):
                future = executor.submit(
                    self._search_options_subset, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
                self.logger.info(f"Restart {i} of {restarts} submitted.")
                futures.add(future)
            self.logger.info(f"Logging Checkpoint 2.")

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    best_cost, selected_options, info = future.result()
                    total_loss_calculations = info['total_loss_calculations']
                    steps = info['steps']
                    # self.levin_loss.cache.update(cache)
                    if best_cost < best_levin_loss_total:
                        best_levin_loss_total = best_cost
                        best_selected_options = selected_options
                    completed += 1
                    # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
                    self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
                    utils.logger_flush(self.logger)
                except Exception as exc:
                    self.logger.error(f'Exception: {exc}')
                    traceback.print_exc()
                    return

        best_selected_options = [all_options[idx] for idx in best_selected_options]
        self.levin_loss.remove_cache()

        # Removing redundant options
        def get_levin_loss(options, trajectories):
            cost = 0
            for problem, trajectory in trajectories.items():
                cost += self.levin_loss.compute_loss_cached(options, 
                                                trajectory, 
                                                problem_str=problem, 
                                                number_actions=3,
                                                cache_enabled=False)[0]
            return cost
    
        best_levin_loss = get_levin_loss(best_selected_options, trajectories)
        
        self.logger.info(f"Levin loss: {best_levin_loss}")
        options = copy.deepcopy(best_selected_options)
        while True and len(options) > 1:
            done = True
            best_loss_so_far = best_levin_loss
            for i in range(len(options)):
                options_cpy = copy.deepcopy(options)
                options_cpy = options_cpy[:i] + options_cpy[i+1:]
                levin_loss = get_levin_loss(options_cpy, trajectories)
                if levin_loss < best_loss_so_far:
                    best_loss_so_far = levin_loss
                    best_options_so_far = options_cpy
                    redundant_idx = i
                    done = False
            if not done:
                best_levin_loss = best_loss_so_far
                options = best_options_so_far
                self.logger.info(f"Levin loss without option #{redundant_idx}: {best_levin_loss}")
            else:
                break
                    


        return list(options)

    def _train_mask_iter(self, trajectories, problem, s, length, agent: PPOAgent):
        sub_trajectory = {problem: trajectories[problem].slice(s, n=length)}
        # learn option with this length
        if self.mask_type != "both":
            if self.mask_type == "internal" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask
            elif self.mask_type == "internal" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_mask_softmax
            elif self.mask_type == "input" and self.mask_transform_type == "quantize":
                mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask
            elif self.mask_type == "input" and self.mask_transform_type == "softmax":
                mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                rollout_func = agent.run_with_input_mask_softmax
            return self._train_mask(mask, agent, sub_trajectory, rollout_func)
        else:
            if self.mask_transform_type == "quantize":
                input_mask = torch.nn.Parameter(torch.randn(agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks
            elif self.mask_transform_type == "softmax":
                input_mask = torch.nn.Parameter(torch.randn(3, agent.observation_space_size), requires_grad=True)
                internal_mask = torch.nn.Parameter(torch.randn(3, self.args.hidden_size), requires_grad=True)
                rollout_func = agent.run_with_both_masks_softmax
            return self._train_mask_both(input_mask, internal_mask, agent, sub_trajectory, rollout_func)

    def _train_mask(self, mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        # mask = torch.nn.Parameter(torch.randn(self.args.hidden_size), requires_grad=True)
        optimizer = torch.optim.Adam([mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                mask_transformed = self._mask_transform_func(mask)

                # Generate a rollout with the mask
                new_trajectory = rollout_func(envs, mask_transformed, trajectory.get_length())

                # Calculate the mask loss            
                # input_probs = torch.nn.functional.log_softmax(torch.stack(new_trajectory.logits), dim=0)
                # target_probs = torch.nn.functional.softmax(torch.stack(trajectory.logits) , dim=0)
                # loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                # mask_loss = loss_fn(input_probs, target_probs)
                loss_fn = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions))
                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_mask = mask_transformed

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_mask = mask_transformed
                optimizer.zero_grad()
                mask_loss.backward()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)
                optimizer.step()

                # Update progress
                steps += 1
            # self.logger.info(f"Steps: {steps}, Best loss: {best_loss}")

        # assert trajectory.get_action_sequence() == new_trajectory.get_action_sequence(), \
        #     f"Mask {mask} does not match the original trajectory for target problem {trajectories.keys()} , \
        #         {new_trajectory.get_action_sequence()}, expected: {trajectory.get_action_sequence()} \n \
        #             primary_problem={agent.extra_info['primary_problem']}"
        trajectory = list(trajectories.values())[0]
        envs = trajectory.get_state_sequence()
        new_trajectory = rollout_func(envs, best_mask, trajectory.get_length())

        applicable = new_trajectory.get_action_sequence() == trajectory.get_action_sequence()
        return best_mask.detach().data, init_loss, best_loss, applicable

    def _train_mask_both(self, input_mask, internal_mask, agent: PPOAgent, trajectories: dict, rollout_func):
        # Initialize the masks as trainable parameters with random values
        input_optimizer = torch.optim.Adam([input_mask], lr=self.args.mask_learning_rate)
        internal_optimizer = torch.optim.Adam([internal_mask], lr=self.args.mask_learning_rate)

        init_loss = None
        best_loss = None
        best_input_mask = None
        best_internal_mask = None

        steps = 0
        # with tqdm(total=self.args.mask_learning_steps, desc="Mask Learning Steps") as pbar:
        while steps < self.args.mask_learning_steps:
            # Iterate over trajectories
            for _, trajectory in trajectories.items():
                if steps >= self.args.mask_learning_steps:
                    break

                # Forward pass with mask
                envs = trajectory.get_state_sequence()
                actions = trajectory.get_action_sequence()
                agent.eval()

                # Apply transformations to the mask
                input_mask_discretized = self._mask_transform_func(input_mask)

                internal_mask_discretized = self._mask_transform_func(internal_mask)

                new_trajectory = rollout_func(envs, input_mask_discretized, internal_mask_discretized, trajectory.get_length())

                loss_fn = torch.nn.CrossEntropyLoss()
                mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions)) 

                if not init_loss:
                    init_loss = mask_loss.item()
                    best_loss = init_loss
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                # Backward pass and optimization
                if mask_loss.item() < best_loss:
                    best_loss = mask_loss.item()
                    best_input_mask = input_mask_discretized
                    best_internal_mask = internal_mask_discretized

                internal_optimizer.zero_grad()
                mask_loss.backward(retain_graph=True)  
                internal_optimizer.step()

                if steps % self.args.input_update_frequency == 0:  
                    input_optimizer.zero_grad()
                    mask_loss.backward()  
                    input_optimizer.step()
                
                # torch.nn.utils.clip_grad_norm_([mask], max_norm=self.args.max_grad_norm)

                # Update progress
                steps += 1
        
        trajectory = list(trajectories.values())[0]
        envs = trajectory.get_state_sequence()
        new_trajectory = rollout_func(envs, best_input_mask, best_internal_mask, trajectory.get_length())

        applicable = new_trajectory.get_action_sequence() == trajectory.get_action_sequence()
        return (best_input_mask.detach().data, best_internal_mask.detach().data), init_loss, mask_loss.item(), applicable




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

    logger.info(f'mask_type="{args.mask_type}", mask_transform_type="{args.mask_transform_type}, selection_type="{args.selection_type}"')

    module_extractor = LearnOptions(args, logger)
    module_extractor.discover()

    module_extractor = WholeDecOption(args, logger)
    module_extractor.discover()

    # evaluate_all_masks_levin_loss(args, logger)
    # hill_climbing_mask_space_training_data()
    # whole_dec_options_training_data_levin_loss(args, logger)
    # hill_climbing_all_segments()
    # learn_options(args, logger)

    logger.info(f"Run id: {run_name}")
    logger.info(f"logs saved at {args.log_path}")


if __name__ == "__main__":
    main()
