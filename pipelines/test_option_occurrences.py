import os, copy
import tyro
from utils import utils
from pipelines.losses import LogitsLossActorCritic, LevinLossActorCritic
from pipelines.option_discovery import load_options
from dataclasses import dataclass
from environments.environments_combogrid_gym import ComboGym


@dataclass
class Args:
    # exp_id: str = "extract_learnOption_ComboGrid_gw5_h64_l10_r400_envsd0,1,2,3"
    exp_id: str = "extract_learnOption_CrossVal_AvgLoss_ComboGrid_gw5_h64_l10_r2000_envsd0,1,2,3_mskTypeboth_mskTransformsoftmax_selectTypelocal_search"
    """The ID of the finished experiment"""
    seed: int = 1
    """run seed"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""

    env_id = "ComboGrid"
    hidden_size = 64
    game_width = 5


def main(args):
    log_path = os.path.join(args.log_path, args.exp_id)
    log_path += f"/occurances"
    logger, args.log_path = utils.get_logger("print_option_occurances_logger", args.log_level, log_path)
    loss = LevinLossActorCritic(logger, "both", "softmax")

    options, trajectories = load_options(args, logger)
    for i,option in enumerate(options):
        print(f"option {i}",option.feature_mask,"\n", option.actor_mask)
    feature_mask = [agent.feature_mask for agent in options]
    actor_masks = [agent.actor_mask for agent in options]
    option_sizes = [agent.option_size for agent in options]
    target_problems = [option.extra_info["target_problem"] for option in options]
    best_loss = loss.compute_loss(feature_mask, actor_masks, options,"", trajectories, 3, option_sizes)
    print("Best loss:", best_loss)
    # feature_mask.pop(4)
    # actor_masks.pop(4)
    # option_sizes.pop(4)
    # options.pop(4)
    # best_loss = loss.compute_loss(feature_mask, actor_masks, options, "TL-BR", trajectories, 3, option_sizes)
    # print("Best loss without option 4:", best_loss)


    loss.print_output_subpolicy_trajectory(options=options, 
                                            trajectories=trajectories, 
                                            logger=logger)
    # output_str = ""
    # for id,option in enumerate(options):
    #     problems = ["TL-BR", "TR-BL", "BR-TL", "BL-TR"]
    #     for problem in problems:
    #         option.eval()
    #         env = ComboGym(rows=5, columns=5, problem=problem)
    #         max_traj_len=0
    #         trajs = [[[] for _ in range(5)] for _ in range(5)]
    #         for i in range(5):
    #             for j in range(5):
    #                 if env._game._matrix_goal[i][j] != 1:
    #                     env.reset(init_loc=(i,j))
    #                     traj = option.run_with_input_mask_softmax(mask_a=option.actor_mask, mask_f=option.feature_mask, envs=copy.deepcopy(env), max_size_sequence=option.option_size)
    #                     trajs[i][j] += traj.get_action_sequence() 
    #                     max_traj_len = max(max_traj_len, (len(traj.get_action_sequence())+1)*3)
    #         output_str += f"Problem: {problem} | Option: {id}\n"
    #         for rows in trajs:
    #             for i, cell in enumerate(rows):
    #                 if len(cell) == 0:
    #                     output_str += f'{str("G").ljust(max_traj_len)}\t'
    #                 else:
    #                     output_str += f'{str(cell).ljust(max_traj_len)}\t'
    #             output_str += "\n"   
    # with open(f"binary/options/selected_options_performance_crossVal.txt", 'w') as f:
    #     f.write(output_str)

    utils.logger_flush(logger)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)