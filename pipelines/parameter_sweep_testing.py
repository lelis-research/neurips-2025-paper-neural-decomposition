import os
from environments.utils import get_single_environment_builder
import torch
import time
import tyro
import copy
import random
import numpy as np
import gymnasium as gym
from utils import utils
from logging import Logger
from typing import Union, List
from agents.policy_guided_agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter
from pipelines.option_discovery import load_options
from dataclasses import dataclass
from training.train_ppo_agent import train_ppo, train_ppo_async
from environments.environments_minigrid import make_env_four_rooms
from environments.environments_combogrid import PROBLEM_NAMES as COMBOGRID_NAMES
from environments.environments_combogrid_gym import make_env as make_env_combogrid


@dataclass
class Args:
    # exp_id: str = "extract_basePolicyTransferred_ComboGrid_gw5_h64_envsd0,1,2,3"
    # exp_id: str = "extract_fineTuning_notFiltered_ComboGrid_gw5_h64_l10_envsd0,1,2,3_selectTypelocal_search_reg0.0maxNumOptions5"
    # exp_id: str = "extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinput_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5"
    # exp_id: str = "extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5"
    exp_id: str = ""
    # exp_id: str = "extract_wholeDecOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5"
    # exp_id: str = "extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinternal_mskTransformsoftmax_selectTypelocal_search_reg0maxNumOptions5"
    """The ID of the finished experiment"""
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    method: str = ""
    """Determines the baseline that is being tested; Choices: ['no_options']"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameter arguments
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3
    """"""
    
    # Testing specific arguments
    test_exp_id: str = ""
    """The ID of the new experiment"""
    # test_exp_name: str = "test_learnOptions_input_filtered"
    # test_exp_name: str = "test_learnOptions_both_filtered"
    # test_exp_name: str = "test_learnOptions_internal_filtered"
    # test_exp_name: str = "test_fine_tuning_unfiltered"
    test_exp_name: str = "test_base_policy_transferred"
    # test_exp_name: str = "test_no_options"
    # test_exp_name: str = "test_wholeDecOption"
    """the name of this experiment"""
    test_env_id: str = "ComboGrid"
    """the id of the environment for testing
    choices from [ComboGrid, MiniGrid-FourRooms-v0]"""
    test_problems: List[str] = tuple()
    """"""
    test_env_seeds: Union[List[int], str] = (14,)
    """the seeds of the environment for testing"""
    total_timesteps: int = 1_000_000
    """total timesteps for testing"""
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 5e-05) # Vanilla RL
    # learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001)
    learning_rate: Union[List[float], float] = (0.0005, 0.001, 0.001)
    # learning_rate: Union[List[float], float] = (0.0005, 0.0005, 0.0005) # Dec-Option Whole 
    """the learning rate of the optimize for testing"""
    num_envs: int = 4
    """the number of parallel game environments for testing"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout for testing"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks for testing"""
    # gamma: float = 0.99
    gamma: float = 1
    """the discount factor gamma for testing"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation for testing"""
    num_minibatches: int = 4
    """the number of mini-batches for testing"""
    update_epochs: int = 4
    """the K epochs to update the policy for testing"""
    norm_adv: bool = True
    """Toggles advantages normalization for testing"""
    # clip_coef: Union[List[float], float] = (0.15, 0.1, 0.2) # Vanilla RL
    # clip_coef: Union[List[float], float] = (0.25, 0.2, 0.2)
    clip_coef: Union[List[float], float] = (0.2, 0.2, 0.2) # Combogrid test
    # clip_coef: Union[List[float], float] = (0.3, 0.25, 0.15) # Dec-Option Whole 
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    # ent_coef: Union[List[float], float] = (0.05, 0.2, 0.0) # Vanilla RL
    # ent_coef: Union[List[float], float] = (0.15, 0.05, 0.05) # Dec-Option Whole 
    # ent_coef: Union[List[float], float] = (0.1, 0.1, 0.1) # ComboGrid
    ent_coef: Union[List[float], float] = (0.2, 0.1, 0.1) # Experimental values
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
    
    # script arguments
    seed: int = 0
    """run seed"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases; Set to true if you want to plot the experiment"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""


def train_ppo_with_options(options: List[PPOAgent], test_exp_id: str, env_seed: int, args: Args, logger: Logger):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    run_name = f"{test_exp_id}_sd{args.seed}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=test_exp_id,
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

    problem = None
    if "ComboGrid" in args.env_id:
       problem = args.test_problem
    # envs = gym.vector.SyncVectorEnv([get_single_environment_builder(args, env_seed, problem, options=options, is_test=True) for _ in range(args.num_envs)])
    envs = gym.vector.AsyncVectorEnv([get_single_environment_builder(args, env_seed, problem, options=options, is_test=False) for _ in range(args.num_envs)],
                                     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP) # TODO: change is_test to True for testing
    
    if args.method == "no_options":
        assert envs.action_space[0].n == args.number_actions, f"no options should have same action space as the environment: {envs.action_space[0].n}"

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    model_path = f'binary/models/parameter_sweep/{test_exp_id}/seed={args.seed}/extended_MODEL.pt'
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{test_exp_id}_trained_with_options/seed={args.seed}/_t{int(time.time())}"
    writer = SummaryWriter(f"outputs/tensorboard/runs/parameter_sweep/{run_name}")
    hyperparameters = dict(vars(args))
    hyperparameters.update({"test_exp_id": test_exp_id})
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparameters.items()])),
    )
    options_info = {f"option{i}":(option.mask, option.option_size, option.problem_id) for i, option in enumerate(options)}
    writer.add_text(
        "options_setting",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in options_info.items()])),)
    logger.info(f"Reporting tensorboard summary writer on outputs/tensorboard/runs/{run_name}")

    if isinstance(envs, gym.vector.SyncVectorEnv):
        train_ppo(envs=envs, 
                seed=env_seed, 
                args=args, 
                model_file_name=model_path, 
                device=device, 
                logger=logger, 
                writer=writer,
                parameter_sweeps=True)
    elif isinstance(envs, gym.vector.AsyncVectorEnv):
        train_ppo_async(envs=envs, 
                    seed=env_seed, 
                    args=args, 
                    model_file_name=model_path, 
                    device=device, 
                    logger=logger, 
                    writer=writer,
                    parameter_sweeps=True)
    
    if args.track:
        wandb.finish()


def main(args: Args):
    # Custom parameters for each problem
    lrs = args.learning_rate
    clip_coef = args.clip_coef
    ent_coef = args.ent_coef
    for i, (problem, seed) in enumerate(zip(args.test_problems, args.test_env_seeds)):
        mod_args = copy.deepcopy(args)
        mod_args.learning_rate = lrs[i]
        mod_args.clip_coef = clip_coef[i]
        mod_args.ent_coef = ent_coef[i]
        test_exp_sub_id = f"_lr{mod_args.learning_rate}_clip{mod_args.clip_coef}_ent{mod_args.ent_coef}_envsd{seed}"
        args.log_path = os.path.join(mod_args.log_path, test_exp_sub_id)
        logger, mod_args.log_path = utils.get_logger("test_options_logger", mod_args.log_level, args.log_path)
        
        logger.info(f"Testing with {mod_args.method} on {problem}, env_seed={seed}")
        logger.info(f"Learning rate: {mod_args.learning_rate}, {type(mod_args.learning_rate)}")
        logger.info(f"Clip coefficient: {mod_args.clip_coef}, {type(mod_args.clip_coef)}")
        logger.info(f"Entropy coefficient: {mod_args.ent_coef}, {type(mod_args.ent_coef)}")

        if mod_args.method == "no_options":
            options = []
        else:
            options, _ = load_options(mod_args, logger)
        for option in options:
            print((option.mask, option.option_size, option.problem_id))

        logger.info(f"Testing by training on {problem}, env_seed={seed}")
        mod_args.batch_size = int(mod_args.num_envs * mod_args.num_steps)
        mod_args.minibatch_size = int(mod_args.batch_size // mod_args.num_minibatches)
        mod_args.num_iterations = mod_args.total_timesteps // mod_args.batch_size
        
        mod_args.test_seed = seed
        mod_args.test_problem = problem
        test_exp_id = f'{mod_args.test_exp_id}_{test_exp_sub_id}'
        train_ppo_with_options(options, test_exp_id, seed, mod_args, logger)
        utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Setting the test experiment id
    if args.test_exp_id == "":
        args.test_exp_id = f'{args.test_exp_name}_{args.test_env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}'
    args.log_path = os.path.join(args.log_path, "param_sweep")
    if args.method == "no_options":
        args.log_path = os.path.join(args.log_path, args.test_exp_id, f"seed={args.seed}")
    else:
        args.log_path = os.path.join(args.log_path, args.exp_id, f"seed={args.seed}", args.test_exp_id)

    # Setting test seeds and test problem names
    if isinstance(args.test_env_seeds, list) or isinstance(args.test_env_seeds, tuple):
        args.test_env_seeds = list(map(int, args.test_env_seeds))
    elif isinstance(args.test_env_seeds, str):
        start, end = map(int, args.test_env_seeds.split(","))
        args.test_env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.test_env_id == "ComboGrid":
        args.test_problems = [COMBOGRID_NAMES[i] for i in args.test_env_seeds]
    elif args.test_env_id == "MiniGrid-FourRooms-v0":
        args.test_problems = [args.test_env_id + str(seed) for seed in args.test_env_seeds]

    if isinstance(args.learning_rate, str):
        args.learning_rate = [float(args.learning_rate)]
    if isinstance(args.clip_coef, str):
        args.clip_coef = [float(args.clip_coef)]
    if isinstance(args.ent_coef, str):
        args.ent_coef = [float(args.ent_coef)]
    
    main(args)