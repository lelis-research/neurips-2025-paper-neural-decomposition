from pipelines.option_discovery import process_args, get_single_environment, load_options
from utils import utils
from agents.policy_guided_agent import PPOAgent
import random
import numpy as np
import torch
import traceback

def regenerate_trajectories(args, verbose=False, logger=None):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'/home/rezaabdz/scratch/binary/models/{model_directory}/seed={args.seed}/ppo_first_MODEL.pt'
        env = get_single_environment(args, seed=seed)
        
        if verbose:
            logger.info(f"Loading Trajectories from {model_path} ...")
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory, info = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

        print(info)

        if verbose:
            logger.info(f"The trajectory length: {len(trajectory.get_state_sequence())}")

    return trajectories


args = process_args()

# args.model_paths = (
#         'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
#         'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
#         'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
#         'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR'
#         )
# args.model_paths = (
#         'train_ppoAgent_ComboGrid_gw6_h64_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
#         'train_ppoAgent_ComboGrid_gw6_h64_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
#         'train_ppoAgent_ComboGrid_gw6_h64_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
#         'train_ppoAgent_ComboGrid_gw6_h64_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR'
        # )
args.model_paths = (
        'train_ppoAgent_ComboGrid_gw6_h64_lr0.0005_clip0.4_ent0.15_envsd0_TL-BR',
        'train_ppoAgent_ComboGrid_gw6_h64_lr0.0005_clip0.4_ent0.15_envsd1_BR-TL',
        'train_ppoAgent_ComboGrid_gw6_h64_lr0.0005_clip0.4_ent0.15_envsd2_BR-TL',
        'train_ppoAgent_ComboGrid_gw6_h64_lr0.0005_clip0.4_ent0.15_envsd3_BL-TR',
        
        )
# args.model_paths = (
#         'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
#         'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
#         'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
#         'train_ppoAgent_ComboGrid_gw6_h6_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
        
#         )
# args.model_paths = (
#     "train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0",
#     "train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1",
#     "train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2",
    
# )

# args.env_id = "MiniGrid-SimpleCrossingS9N1-v0"
args.env_id = "ComboGrid"
args.env_seeds = (0, 1, 2, 3)
# args.env_seeds = (0, 1, 2)
args.game_width = 6
args.env_seeds = [int(s) for s in args.env_seeds]


lengths = {}
logger, _ = utils.get_logger('test_primary_model', args.log_level, args.log_path)

for i in range(0,14):
    args.seed = i
    try:
        trajectories = regenerate_trajectories(args, verbose=False, logger=None)
    except:
        traceback.print_exc()
    for problem, t in trajectories.items():
        if problem not in lengths:
            lengths[problem] = []
        lengths[problem].append(t.get_length())


for p, ls in lengths.items():
    print(f"problem: {p}")
    for i, l in enumerate(ls):
        print(f"seed={i}, l={l}")