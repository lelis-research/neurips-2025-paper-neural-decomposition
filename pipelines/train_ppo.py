import os
import random
import time
from environments.utils import get_single_environment_builder
import torch
import tyro
import numpy as np
import gymnasium as gym
from utils import utils
from typing import Union, List, Tuple
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from environments.environments_combogrid_gym import make_env as make_env_combogrid
from environments.environments_combogrid import PROBLEM_NAMES as COMBOGRID_PROBLEMS
from environments.environments_minigrid import make_env_simple_crossing, make_env_four_rooms
from pipelines.option_discovery import load_options
from training.train_ppo_agent import train_ppo


@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppoAgent"
    """the name of this experiment"""
    env_id: str = "MiniGrid-MultiRoom-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0, MiniGrid-FourRooms-v0, MiniGrid-Unlock-v0, MiniGrid-MultiRoom-v0]
    """
    method: str = "options"
    option_mode: str = "didec"
    # env_seeds: Union[List[int], str] = (0,1,2) # SimpleCrossing
    # env_seeds: Union[List, str, Tuple] = (0,1,2,3) # ComboGrid
    # env_seeds: Union[List[int], str] = (8,41,51) # FourRooms
    # env_seeds: Union[List[int], str] = (1,3,17) # Unlock
    env_seeds: Union[List[int], str] = (230, 431) # MultiRoom Unlock
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'.
    This determines the exact environments that will be separately used for training.
    """
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_tensorboard: bool = False
    """if toggled, this experiment will be tracked with Tensorboard SummaryWriter"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    cpus: int = 0
    """"Not used in this experiment"""
    save_run_info: int = 0
    
    # hyperparameter arguments
    game_width: int = 9
    """the length of the combo/mini-grid square"""
    hidden_size: int = 64
    # hidden_size: int = 6
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3
    # number_actions: int = 6 #Unlock and MultiRoom Unlock

    # Specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps for testinging"""
    # learning_rate: Union[Tuple[float, ...], float] = (2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4) # ComboGrid
    # learning_rate: Union[List[float], float] = (0.005, 0.0005, 0.001) # Vanilla RL FourRooms
    # learning_rate: Union[List[float], float] = (0.005, 0.0005, 0.005) # Vanilla RL FourRooms View 5
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.001) # Didec RL FourRooms no reg
    # learning_rate: Union[List[float], float] = (0.005, 0.001, 0.005) # Didec RL FourRooms View 5
    # learning_rate: Union[List[float], float] = (0.001, 0.005, 0.001) # Didec-reg RL FourRooms
    # learning_rate: Union[List[float], float] = (0.005, 0.001, 0.005) # Didec-reg RL FourRooms View 5
    # learning_rate: Union[List[float], float] = (0.0005, 0.005, 0.005) #dec-whole FourRooms
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.005) #dec-whole FourRooms View 5
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.001) #fine-tune FourRooms
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.005) #fine-tune FourRooms View 5
    # learning_rate: Union[List[float], float] = (0.001, 0.005, 0.0005) #neural-augmented FourRooms
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.0005) #neural-augmented FourRooms View 5
    # learning_rate: Union[List[float], float] = (2.5e-4, 2.5e-4, 1e-4) # SimpleCrossing
    # learning_rate: Union[List[float], float] = (0.005, 0.005, 0.005) # Unlock
    learning_rate: Union[List[float], float] = (0.005, 0.0005) # MultiRoom Didec
    # learning_rate: Union[List[float], float] = (0.005, 0.01) # MultiRoom Didec-reg
    # learning_rate: Union[List[float], float] = (0.001, 0.005) # MultiRoom Dec-whole
    # learning_rate: Union[List[float], float] = (0.005, 0.005) # MultiRoom fine tune
    # learning_rate: Union[List[float], float] = (0.005, 0.001) # MultiRoom vanilla
    # learning_rate: Union[List[float], float] = (0.001, 0.001) # MultiRoom neural augmented
    """the learning rate of the optimize for testinging"""
    num_envs: int = 4
    """the number of parallel game environments for testinging"""
    num_steps: int = 722
    """the number of steps to run in each environment per policy rollout for testinging"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for testinging"""
    gamma: float = 0.99
    """the discount factor gamma for testinging"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for testinging"""
    num_minibatches: int = 4
    """the number of mini-batches for testinging"""
    update_epochs: int = 4
    """the K epochs to update the policy for testinging"""
    norm_adv: bool = True
    """Toggles advantages normalization for testinging"""
    # clip_coef: Union[Tuple[float, ...], float] = (0.2, 0.2, 0.2, 0.2) # ComboGrid
    # clip_coef: Union[List[float], float] = (0.1, 0.15, 0.15) # Vanilla RL FourRooms
    # clip_coef: Union[List[float], float] = (0.2, 0.15, 0.3) # Vanilla RL FourRooms View 5
    # clip_coef: Union[List[float], float] = (0.3, 0.3, 0.3) # Didec RL FourRooms no reg
    # clip_coef: Union[List[float], float] = (0.3, 0.3, 0.3) # Didec RL FourRooms no reg View 5
    # clip_coef: Union[List[float], float] = (0.2, 0.3, 0.3) # Didec-reg RL FourRooms
    # clip_coef: Union[List[float], float] = (0.2, 0.3, 0.3) # Didec-reg RL FourRooms View 5
    # clip_coef: Union[List[float], float] = (0.3, 0.3, 0.3) # dec-whole FourRooms
    # clip_coef: Union[List[float], float] = (0.2, 0.3, 0.2) # dec-whole FourRooms View 5
    # clip_coef: Union[List[float], float] = (0.3, 0.3, 0.3) # fine-tune FourRooms
    # clip_coef: Union[List[float], float] = (0.3, 0.3, 0.2) # fine-tune FourRooms View 5
    # clip_coef: Union[List[float], float] = (0.2, 0.3, 0.2) # neural-augmented FourRooms
    # clip_coef: Union[List[float], float] = (0.3, 0.15, 0.3) # neural-augmented FourRooms View 5
    # clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2) # SimpleCrossing
    # clip_coef: Union[List[float], float] = (0.3, 0.2, 0.2) # Unlock
    clip_coef: Union[List[float], float] = (0.3, 0.3) # MultiRoom Didec
    # clip_coef: Union[List[float], float] = (0.3, 0.2) # MultiRoom Didec-reg
    # clip_coef: Union[List[float], float] = (0.2, 0.15) # MultiRoom Vanilla
    # clip_coef: Union[List[float], float] = (0.3, 0.3) # MultiRoom Fine-tune
    # clip_coef: Union[List[float], float] = (0.15, 0.05) # MultiRoom Dec-whole
    # clip_coef: Union[List[float], float] = (0.3, 0.2) # MultiRoom Neural Augmented
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef:Union[Tuple[float, ...], float] = (0.01, 0.01, 0.01, .01) # ComboGrid
    # ent_coef: Union[List[float], float] = (0.01, 0.01, 0.2) # Vanilla RL FourRooms
    # ent_coef: Union[List[float], float] = (0.03, 0.02, 0.05) # Vanilla RL FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.01, 0.03, 0.05) # Didec RL FourRooms no reg
    # ent_coef: Union[List[float], float] = (0.01, 0.05, 0.02) # Didec RL FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.02, 0.05, 0.03) # Didec-reg FourRooms
    # ent_coef: Union[List[float], float] = (0.1, 0.05, 0.01) # Didec-reg FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.03, 0.03, 0.01) # dec-whole FourRooms
    # ent_coef: Union[List[float], float] = (0.05, 0.1, 0.02) # dec-whole FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.01, 0.01, 0.05) # fine-tune FourRooms
    # ent_coef: Union[List[float], float] = (0.01, 0.02, 0.01) # fine-tune FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.02, 0.02, 0.01) # neural-augmented FourRooms
    # ent_coef: Union[List[float], float] = (0.01, 0.02, 0.03) # neural-augmented FourRooms View 5
    # ent_coef: Union[List[float], float] = (0.02, 0.02, 0.015) # SimpleCrossing
    # ent_coef: Union[List[float], float] = (0.05, 0.05, 0.2) # Unlock
    ent_coef: Union[List[float], float] = (0.02, 0.03) # MultiRoom Didec
    # ent_coef: Union[List[float], float] = (0.01, 0.1) # MultiRoom Didec-reg
    # ent_coef: Union[List[float], float] = (0.2, 0.03) # MultiRoom Vanilla
    # ent_coef: Union[List[float], float] = (0.02, 0.01) # MultiRoom Fine-tune
    # ent_coef: Union[List[float], float] = (0.01, 0.01) # MultiRoom Neural Augmented
    # ent_coef: Union[List[float], float] = (0.03, 0.01) # MultiRoom Dec-whole
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    sweep_run: int = 0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    env_seed: int = -1
    """the seed of the environment (set in runtime)"""
    seed: int = 12
    """experiment randomness seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    models_path_prefix: str = "binary/models"
    
    log_level: str = "INFO"
    """The logging level"""


@utils.timing_decorator
def main(args: Args):
    
    run_name = f"{args.exp_id}_sd{args.seed}_training_t{int(time.time())}" 
    run_index = f"train_ppo_t{int(time.time())}" 
    
    log_path = os.path.join(args.log_path, args.exp_id, f"seed={args.seed}", "train_ppo")

    logger, _ = utils.get_logger('ppo_trainer_logger_' + str(args.env_seed) + "_" + args.exp_name, args.log_level, log_path)

    logger.info(f"\n\nExperiment: {args.exp_id}\n\n")

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
    
    writer = None
    if args.track_tensorboard:
        # Setting up tensorboard summary writer
        writer_path = f"{args.exp_id}/seed={args.seed}/{run_index}"
        writer = SummaryWriter(f"outputs/tensorboard/runs/{writer_path}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger.info(f"Constructing tensorboard summary writer on outputs/tensorboard/runs/{run_name}")
        

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Parameter logging
    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    # Environment creation
    problem = None
    options = None
    if args.method != "no_options":
        options, _ = load_options(args, logger)
    if "ComboGrid" in args.env_id:
        problem = args.problem
    envs = gym.vector.SyncVectorEnv([get_single_environment_builder(args, args.env_seed, problem, is_test=False, options=options) for _ in range(args.num_envs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    
    # model_path = f'{args.models_path_prefix}/{args.exp_id}/seed={args.seed}/ppo_first_MODEL.pt'
    if args.sweep_run == 1:
        model_path = f'{args.models_path_prefix}/{args.env_id}_width={args.game_width}_{args.option_mode}/seed={args.seed}/{args.env_id.lower()}-{COMBOGRID_PROBLEMS[args.env_seed] if args.env_id == "ComboGrid" else args.env_seed}-3.pt'
    else:
        model_path = f'binary/models_sweep_{args.env_id}_{args.env_seed}_{args.option_mode}/seed={args.seed}/{args.exp_id}.pt'
        if os.path.isfile(model_path):
            logger.info(f"Model already exists. Stopping training...")
            exit()

    train_ppo(envs=envs, 
              seed=args.env_seed, 
              args=args, 
              model_file_name=model_path, 
              device=device, 
              logger=logger, 
              writer=writer)
    if args.track:
        wandb.finish()
    # wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)

    env_idx = args.seed % len(args.env_seeds)
    args.seed = int(args.seed // len(args.env_seeds))

    if args.method == "no_options":
        args.option_mode = "vanilla"

    # Setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}'
    
    # Processing seeds from arguments
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        args.env_seeds = []
        seed_intervals = args.env_seeds.split(",")
        for interval in seed_intervals:
            bounds = list(map(int, interval.split("-")))
            if len(bounds) != 2:
                bounds.append(bounds[0])
            start, end = bounds
            args.env_seeds += list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    # Parameter specification for each problem
    if args.sweep_run == 1:
        ent_coefs = {
            "didec": (0.02, 0.03),
            "didec-reg": (0.02, 0.03),
            "vanilla": (0.01, 0.01), # MultiRoom Vanilla
            "fine-tune": (0.03, 0.03), # MultiRoom Fine-tune
            "neural-augmented": (0.02, 0.01), # MultiRoom Neural Augmented
            "dec-whole": (0.03, 0.01) # MultiRoom Dec-whole
        }

        learning_rates = {
            "didec": (0.005, 0.005),
            "didec-reg": (0.01, 0.005),
            "vanilla": (0.001, 0.001), # MultiRoom Vanilla
            "fine-tune": (0.01, 0.01), # MultiRoom Fine-tune
            "neural-augmented": (0.005, 0.0005), # MultiRoom Neural Augmented
            "dec-whole": (0.0005, 0.0005) # MultiRoom Dec-whole
        }
        clip_coefs = {
            "didec": (0.3, 0.3),
            "didec-reg": (0.3, 0.2),
            "vanilla": (0.3, 0.3), # MultiRoom Vanilla
            "fine-tune": (0.3, 0.3), # MultiRoom Fine-tune
            "neural-augmented": (0.15, 0.15), # MultiRoom Neural Augmented
            "dec-whole": (0.1, 0.2) # MultiRoom Dec-whole
        }
    
        lrs = learning_rates[args.option_mode]
        clip_coef = clip_coefs[args.option_mode]
        ent_coef = ent_coefs[args.option_mode]
    else:
        lrs = args.learning_rate
        clip_coef = args.clip_coef
        ent_coef = args.ent_coef
    exp_id = args.exp_id
    if isinstance(lrs, float) or len(lrs) == 1:
        lrs = tuple(lrs) * len(args.env_seeds)
    if isinstance(clip_coef, float) or len(clip_coef) == 1:
        clip_coef = tuple(clip_coef) * len(args.env_seeds)
    if isinstance(ent_coef, float) or len(ent_coef) == 1:
        ent_coef = tuple(ent_coef) * len(args.env_seeds)
    
    for i in range(len(args.env_seeds)):
        if i != env_idx:
            continue
        args.env_seed = args.env_seeds[i]
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        args.ent_coef = ent_coef[i]
        args.clip_coef = clip_coef[i]
        args.learning_rate = lrs[i]
        args.exp_id = f'{exp_id}_lr{args.learning_rate}_clip{args.clip_coef}_ent{args.ent_coef}_envsd{args.env_seed}'
        if args.env_id == "ComboGrid":
            args.problem = COMBOGRID_PROBLEMS[args.env_seed]
            args.exp_id += f'_{args.problem}'
        main(args)
    