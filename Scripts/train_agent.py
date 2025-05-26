import numpy as np
import torch
import os
import random
import pickle
import multiprocessing
from torch.utils.tensorboard import SummaryWriter

from Agents.PPOAgent import PPOAgent
from Agents.ElitePPOAgent import ElitePPOAgent
from Agents.RandomAgent import RandomAgent
from Agents.SACAgent import SACAgent
from Agents.DDPGAgent import DDPGAgent
from Agents.DQNAgent import DQNAgent
from Agents.NStepDQNAgent import NStepDQNAgent
from Agents.A2CAgent import A2CAgent
# from Agents.PPOAgentRNN import PPOAgentRNN    
# from Agents.RandomAgentRNN import RandomAgent
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
                  render_mode=args.training_render_mode,
                  max_steps=args.training_env_max_steps,
                  )
    print(f"Obs Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    if args.agent_class == "PPOAgent":
        keys = ["gamma", "lamda",
                    "epochs", "total_steps", "rollout_steps", "num_minibatches",
                    "flag_anneal_step_size", "step_size",
                    "entropy_coef", "critic_coef",  "clip_ratio", 
                    "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
                    "flag_anneal_var", "var_coef", "l1_lambda",
                    ]
    elif args.agent_class in ["DQNAgent", "NStepDQNAgent"]:
        keys = ["gamma", "step_size",
                    "batch_size", "target_update_freq",
                    "epsilon", "replay_buffer_cap",
                    "action_res"]
    elif args.agent_class == "DDPGAgent":
        keys = ["gamma", "tau", 
                "actor_lr", "critic_lr",
                "buf_size", "batch_size",
                "noise_phi", "ou_theta", "ou_sigma",
                "epsilon_end", "decay_steps"]
    elif args.agent_class == "A2CAgent":
        keys = ["gamma", "step_size", "rollout_steps", "lamda"]
    else:
        raise NotImplementedError("Agent class not known")
    
    agent_kwargs = {k: getattr(args, k) for k in keys}
    agent_class = eval(args.agent_class)

    if args.load_agent is None:
        print("Training a new agent")
        agent = agent_class(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space,
                        env.single_action_space if hasattr(env, "single_action_space") else env.action_space,
                        device=args.device,
                        **agent_kwargs
                        )
    else:
        agent_path = os.path.join(args.res_dir, args.load_agent, "best.pt")
        print(f"Loading agent from {agent_path}")
        agent = agent_class.load(agent_path)
        agent.initialize_params(**agent_kwargs)
    

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
            print(f"Experiment directory {exp_dir} already exists !")
        writer = SummaryWriter(log_dir=exp_dir)
    
    if args.exp_total_steps > 0 and args.exp_total_episodes == 0:
        result, best_agent = agent_environment_step_loop(env, agent, args.exp_total_steps, writer=writer, save_frame_freq=args.save_frame_freq)
    elif args.exp_total_episodes > 0 and args.exp_total_steps == 0:
        result, best_agent = agent_environment_episode_loop(env, agent, args.exp_total_episodes, writer=writer, save_frame_freq=args.save_frame_freq)
    else:
        raise ValueError("Both steps and episodes are greater than 0")

    if args.save_results:
        with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
            pickle.dump(result, f)
        agent.save(os.path.join(exp_dir, "final.pt"))
        best_agent.save(os.path.join(exp_dir, "best.pt"))
        writer.close()      
    env.close()
    return result, best_agent



def train_parallel_seeds(seeds, args):
    # cap the number of workers to at most len(seeds) or the cpu count
    num_workers = min(args.num_workers, len(seeds), multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=num_workers)
    try:
        results_bestagents = pool.starmap(train_single_seed, [(seed, args) for seed in seeds], chunksize=1)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating all processes…")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()

    results = [x[0] for x in results_bestagents]
    best_agents = [x[1] for x in results_bestagents]
    return results, best_agents

    

    
