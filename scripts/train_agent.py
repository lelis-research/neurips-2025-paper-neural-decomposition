import numpy as np
import torch
import os
import random
import pickle
import multiprocessing
from torch.utils.tensorboard import SummaryWriter

from Agents.PPOAgent import PPOAgent
from Agents.RandomAgent import RandomAgent
from Environments.GetEnvironment import get_env
from Experiments.EnvAgentLoops import agent_environment_step_loop, agent_environment_episode_loop

    
def train_single_seed(seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = get_env(env_name=args.training_env_name,
                  env_params=args.training_env_params,
                  wrapping_lst=args.training_env_wrappers,
                  wrapping_params=args.training_wrapping_params,
                  render_mode=args.training_render_mode
                  )
    
    ppo_keys = ["gamma", "lamda",
                "epochs", "total_steps", "rollout_steps", "num_minibatches",
                "flag_anneal_step_size", "step_size",
                "entropy_coef", "critic_coef",  "clip_ratio", 
                "flag_clip_vloss", "flag_norm_adv", "max_grad_norm"
                ]
    agent_kwargs = {k: getattr(args, k) for k in ppo_keys}

    agent = PPOAgent(env.observation_space, 
                     env.action_space,
                     device=torch.device("cpu"),
                     **agent_kwargs
                     )
    # agent = RandomAgent(env.observation_space, 
    #                     env.action_space,
    #                     )
    writer = None
    if args.save_results:
        if args.exp_total_steps > 0 and args.exp_total_episodes == 0: 
            exp_dir = os.path.join(args.res_dir, f"{args.training_env_name}_{seed}_{args.exp_total_steps}_{args.nametag}")
        elif args.exp_total_episodes > 0 and args.exp_total_steps == 0:
            exp_dir = os.path.join(args.res_dir, f"{args.training_env_name}_{seed}_{args.exp_total_episodes}_{args.nametag}")
        else:
            raise ValueError("Both steps and episodes are greater than 0")
        
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError(f"Experiment directory {exp_dir} already exists. Please choose a different name or remove the existing directory.")
        writer = SummaryWriter(log_dir=exp_dir)
    
    if args.exp_total_steps > 0 and args.exp_total_episodes == 0:
        result = agent_environment_step_loop(env, agent, args.exp_total_steps, writer=writer, save_frame_freq=args.save_frame_freq)
    elif args.exp_total_episodes > 0 and args.exp_total_steps == 0:
        result = agent_environment_episode_loop(env, agent, args.exp_total_episodes, writer=writer, save_frame_freq=args.save_frame_freq)
    else:
        raise ValueError("Both steps and episodes are greater than 0")

    if args.save_results:
        with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
            pickle.dump(result, f)
        agent.save(os.path.join(exp_dir, "final.pt"))
    env.close()
    return result


def train_parallel_seeds(seeds, args):
    # Use multiprocessing to run experiments in parallel.
    pool = multiprocessing.Pool(processes=len(seeds))
    try:
        results = pool.starmap(train_single_seed, [(seed, args) for seed in seeds])
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating all processes...")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()

    return results


    

    
