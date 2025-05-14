import os
import sys
sys.path.append("C:\\Users\\Parnian\\Projects\\neurips-2025-paper-neural-decomposition")
sys.path.append("/home/iprnb/scratch/neurips-2025-paper-neural-decomposition")
import random
import time
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
from environments.environments_minigrid import make_env_simple_crossing, make_env_four_rooms, make_env_unlock, make_env_multiroom
from training.train_ppo_agent import train_ppo
from option_discovery import load_options


@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppoAgent_ComboGrid_Primary"
    """the name of this experiment"""
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, SimpleCrossing, FourRooms, Unlock, MultiRoom]
    """
    # env_seeds: Union[List[int], str] = (0,1,2) # SimpleCrossing
    # env_seeds: int = 12 # ComboGrid
    # env_seeds: Union[List[int], str] = (41,51,8) # FourRooms
    # env_seeds: Union[List[int], str] = (8,) # FourRooms
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'.
    This determines the exact environments that will be separately used for training.
    """
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_tensorboard: bool = False
    """if toggled, this experiment will be tracked with Tensorboard SummaryWriter"""
    wandb_project_name: str = "BASELINE1_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    cpus: int = 0
    """"Not used in this experiment"""
    
    # hyperparameter arguments
    game_width: int = 3
    """the length of the combo/mini-grid square"""
    max_episode_length: int = 30
    """"""
    visitation_bonus: int = 0
    """"""
    use_options: int = 0
    """"""
    processed_options: int = 1
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
    sweep_early_stop: int = 0
    """"""
    entropy_threshold: float = 0.2
    """"""
    return_threshold: int = 10
    """"""
    exp_mode: str = None

    # Specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps for testinging"""
    learning_rate: float = 5e-4 # ComboGrid
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL FourRooms
    # learning_rate: Union[List[float], float] = (5e-05,) # Vanilla RL FourRooms
    # learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001) # SimpleCrossing
    actor_lr: float = 0.0005
    critic_lr: float = 0.0005
    """the learning rate of the optimize for testing"""
    num_envs: int = 8
    """the number of parallel game environments for testing"""
    num_steps: int = 60
    """the number of steps to run in each environment per policy rollout for testing"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for testing"""
    anneal_entropy: int = 0
    """Toggle entropy coefficient annealing"""
    gamma: float = 0.99
    """the discount factor gamma for testing"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for testing"""
    num_minibatches: int = 4
    """the number of mini-batches for testing"""
    update_epochs: int = 8
    """the K epochs to update the policy for testing"""
    norm_adv: bool = True
    """Toggles advantages normalization for testing"""
    clip_coef: float = 0.15 # ComboGrid
    # clip_coef: Union[List[float], float] = (0.15, 0.1, 0.2) # Vanilla RL FourRooms
    # clip_coef: Union[List[float], float] = (0.2,) # Vanilla RL FourRooms
    # clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2) # SimpleCrossing
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02 # ComboGrid
    # ent_coef: Union[List[float], float] = (0.05, 0.2, 0.0) # Vanilla RL FourRooms
    # ent_coef: Union[List[float], float] = (0.1, 0.1, 0.1) # SimpleCrossing
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    env_seed: int = 12
    """the seed of the environment (set in runtime)"""
    seed: int = 23
    """experiment randomness seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    
    log_level: str = "INFO"
    """The logging level"""


@utils.timing_decorator
def main(args: Args):
    
    run_name = f"{args.exp_id}_sd{args.seed}_training_t{int(time.time())}" 
    run_index = f"train_ppo_t{int(time.time())}" 
    
    log_path = os.path.join(args.log_path, args.exp_id, f"seed={args.seed}", "train_ppo")

    logger, _ = utils.get_logger('ppo_trainer_logger_' + str(args.env_seed) + "_" + args.exp_name, args.log_level, log_path)

    logger.info(f"\n\nExperiment: {args.exp_id}\n\n")


    options = None
    if args.use_options == 1:
        if args.env_id == "FourRooms":
            options = {
            'option_folder': f"selected_options/SimpleCrossing",
            'seed': args.seed,
            'env_id': args.env_id,
            'game_width': args.game_width
            }
        elif args.env_id == "ComboGrid":
            options = {
            'option_folder': f"selected_options/ComboGrid",
            'seed': args.seed,
            'env_id': args.env_id,
            'game_width': args.game_width
            }
        elif args.env_id == "MultiRoom":
            options = {
            'option_folder': f"selected_options/Unlock",
            'seed': args.seed,
            'env_id': args.env_id,
            'game_width': args.game_width
            }
            
        

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group="Fixed_str",
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

    visitation_bonus = True if args.visitation_bonus == 1 else False

    # Environment creation
    if args.env_id == "SimpleCrossing":
        envs = gym.vector.AsyncVectorEnv( 
            [make_env_simple_crossing(max_episode_steps=args.max_episode_length, view_size=args.view_size, seed=args.env_seed, visitation_bonus=args.visitation_bonus, options=options) for _ in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
    elif "ComboGrid" in args.env_id:
        problem = args.problem
        envs = gym.vector.AsyncVectorEnv(
            [make_env_combogrid(rows=args.game_width, columns=args.game_width, problem=problem, max_length=args.max_episode_length, visitation_bonus=visitation_bonus, options=options) for _ in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )    
    elif args.env_id == "FourRooms":
        envs = gym.vector.AsyncVectorEnv( 
            [make_env_four_rooms(max_episode_steps=args.max_episode_length, view_size=args.view_size, seed=args.env_seed, visitation_bonus=args.visitation_bonus, options=options) for _ in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
    elif args.env_id == "Unlock":
        envs = gym.vector.AsyncVectorEnv(
            [make_env_unlock(max_episode_steps=args.max_episode_length, view_size=args.view_size, seed=args.env_seed, visitation_bonus=args.visitation_bonus, n_discrete_actions=args.number_actions) for _ in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
    elif args.env_id == "MultiRoom":
        envs = gym.vector.AsyncVectorEnv( 
            [make_env_multiroom(max_episode_steps=args.max_episode_length, view_size=args.view_size, seed=args.env_seed, visitation_bonus=args.visitation_bonus, n_discrete_actions=args.number_actions, options=options) for _ in range(args.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
    else:
        raise NotImplementedError
    
    model_path = f'binary/models_sweep_{args.env_id}_{args.env_seed}_{args.use_options}/seed={args.seed}/{args.exp_id}.pt'
    # model_path = f'binary/models/{args.env_id}/width={args.game_width}/seed={args.seed}/{args.env_id.lower()}-{COMBOGRID_PROBLEMS[args.env_seed] if args.env_id == "ComboGrid" else args.env_seed}-{args.seed}.pt'

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


    # Setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}_option{args.use_options}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_actor-lr{args.actor_lr}_critic-lr{args.critic_lr}' +\
        f'_ent-coef{args.ent_coef}_clip-coef{args.clip_coef}_visit-bonus{args.visitation_bonus}' +\
        f'_ep-len{args.max_episode_length}-ent_an{args.anneal_entropy}-gae{args.gae_lambda}'
    
    
    # Parameter specification for each problem
    args.number_actions = 5 if (args.env_id == "Unlock" or args.env_id == "MultiRoom") else 3
    args.num_steps = args.max_episode_length * 2
    args.view_size = 5 if (args.env_id == "SimpleCrossing" or args.env_id == "FourRooms") else 3
    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    exp_id = args.exp_id
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.env_id == "ComboGrid":
        args.problem = COMBOGRID_PROBLEMS[args.env_seed]
    main(args)
    