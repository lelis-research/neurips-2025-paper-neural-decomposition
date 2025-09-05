import torch
import numpy
from environments.environments_minigrid import get_simplecross_env, get_fourrooms_env
from pipelines.option_discovery import load_options
import gymnasium as gym
from typing import Union, List, Tuple
from dataclasses import dataclass
import tyro
from utils import utils
from environments.utils import get_single_environment_builder
import os
from agents.policy_guided_agent import PPOAgent

@dataclass
class Args:
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    exp_name: str = "train_ppoAgent"
    """the name of this experiment"""
    env_id: str = "MiniGrid-FourRooms-v0"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0, MiniGrid-FourRooms-v0, MiniGrid-Unlock-v0, MiniGrid-MultiRoom-v0]
    """
    method: str = "options"
    option_mode: str = "dec-whole"
    # env_seeds: Union[List[int], str] = (0,1,2) # SimpleCrossing
    # env_seeds: Union[List, str, Tuple] = (0,1,2,3) # ComboGrid
    env_seeds: Union[List[int], str] = (8,51) # FourRooms
    # env_seeds: Union[List[int], str] = (1,3,17) # Unlock
    # env_seeds: Union[List[int], str] = (230, 431) # MultiRoom Unlock
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
    """"""
    reg_coef: Union[List[float], float] = 0.1
    """"""
    mask_type: str = "internal"
    
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
    learning_rate: Union[List[float], float] = (0.005, 0.0005) # MultiRoom Didec
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
    clip_coef: Union[List[float], float] = (0.3, 0.3) # MultiRoom Didec
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: Union[List[float], float] = (0.02, 0.03) # MultiRoom Didec
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    sweep_run: int = 1

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    env_seed: int = -1
    """the seed of the environment (set in runtime)"""
    seed: int = 0
    """experiment randomness seed (set in runtime)"""
    problem: str = ""
    """"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    models_path_prefix: str = "binary/models"
    
    log_level: str = "INFO"
    """The logging level"""

args = tyro.cli(Args)

env_idx = args.seed % len(args.env_seeds)
args.env_seed = args.env_seeds[env_idx]
failed_counts = {}
for option_mode in ["neural-augmented", "dec-whole", "fine-tune", "vanilla"]:
    failed_counts[option_mode] = 0
for seed in range(0,29):
    if seed in [15,19,26]: continue
    args.seed = seed
    print("-"*20, args.seed,"-"*20)
    log_path = os.path.join(args.log_path, args.exp_id, f"seed={args.seed}", "train_ppo")

    logger, _ = utils.get_logger('ppo_trainer_logger_' + str(args.env_seed) + "_" + args.exp_name, args.log_level, log_path)

    for option_mode in ["neural-augmented", "dec-whole", "fine-tune", "vanilla"]:
        args.option_mode = option_mode
        if option_mode == 'vanilla':
            options = None
        else:
            options, _ = load_options(args, logger)

        model_path = f"C:\\Users\\Parnian\\Projects\\neurips-2025-paper-neural-decomposition\\binary\\models\\MiniGrid-FourRooms-v0_width=9_{option_mode}\\seed={args.seed}\\minigrid-fourrooms-v0-8.pt"
        log = torch.load(model_path, map_location='cpu', weights_only=False)

        envs = gym.vector.SyncVectorEnv([get_single_environment_builder(args, args.env_seed, None, is_test=False, options=options) for _ in range(args.num_envs)],
                                            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

        device = torch.device("cpu")
        agent = PPOAgent(envs, hidden_size=args.hidden_size).to(device)
        agent.load_state_dict(log['state_dict'])
        agent.eval()
        print("\n***** ", option_mode," *****\n")
        env = get_fourrooms_env(seed=args.env_seed, options=options)
        trajectory, infos = agent.run(env, 100, deterministic=True)
        if len(trajectory.get_action_sequence())<100:
            print()
            print(trajectory.get_action_sequence())
        else:
            print(f"\n{option_mode} COULD NOT SOLVE SEED {args.seed}")
            failed_counts[option_mode] = failed_counts[option_mode]+1
        
print(failed_counts)
