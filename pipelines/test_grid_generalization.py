import os
import tyro
from utils import utils
from typing import Union, List, Tuple
from pipelines.losses import LevinLossActorCritic
from pipelines.option_discovery import load_options
from dataclasses import dataclass
from environments.environments_combogrid import PROBLEM_NAMES as COMMBOGRID_NAMES, DIRECTIONS
from environments.utils import get_single_environment
import torch


@dataclass
class Args:
    # exp_id: str = "extract_learnOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeinput_mskTransformsoftmax_selectTypelocal_search"
    # exp_id: str = "extract_learnOption_unfiltered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search_reg0"
    exp_id: str = "extract_learnOption_filtered_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search_reg0"
    """The ID of the finished experiment"""
    # env_id: str = "MiniGrid-SimpleCrossingS9N1-v0"
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    game_width: int = 10
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    problems: List[str] = tuple()
    """"""
    env_seeds: Union[Tuple[int, ...], str] = (12,)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""

    # model_paths: List[str] = (
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
    #     'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    # )
    
    # script arguments
    seed: int = 54
    """run seed"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""
    option_mode = "neural-augmented"
    mask_type = "both"
    reg_coef = "0.0"


def main(args: Args):
    log_path = os.path.join(args.log_path, args.exp_id)
    logger, args.log_path = utils.get_logger("test_grid_generalization_logger", args.log_level, log_path)
    
    options, trajectories = load_options(args, logger)
    
    env = get_single_environment(args, args.seed, args.problems[0], True, options)
    
    total_str = ""
    for action in range(len(options)):
        total_str += f"Option #{action}: \n\n"
        for i in range(10):
            for j in range(10):
                action_str = ""
                env.reset(init_loc=(i,j))
                _,_,_,_,info = env.step(3+action)
                actions = info["performed_actions"]
                x = 0
                while x < len(actions):
                    if tuple(actions[x:x+4]) in DIRECTIONS:
                        action_str += DIRECTIONS[tuple(actions[x:x+4])]
                        action_str += " "
                        x += 4
                    else:
                        action_str += f"{actions[x]} "
                        x += 1

                total_str += action_str.center(20)
            total_str += "\n"
        total_str += "-"*60
    with open("options_neural_aug.txt", "w") as f:
        f.write(total_str)
            



if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.env_id == "ComboGrid":
        args.problems = [COMMBOGRID_NAMES[i] for i in args.env_seeds]
    main(args)