import numpy as np
import torch
import random
import os
import pickle

from Agents.PPOAgent import PPOAgent
from Agents.RandomAgent import RandomAgent

from Environments.GetEnvironment import get_env
from Experiments.EnvAgentLoops import agent_environment_step_loop, agent_environment_episode_loop

def test_agent(seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = get_env(env_name=args.test_env_name,
                  env_params=args.test_env_params,
                  wrapping_lst=args.test_env_wrappers,
                  wrapping_params=args.test_wrapping_params,
                  render_mode="human")
    
    agent_path = os.path.join(args.res_dir, args.test_agent_path, "final.pt")
    agent = PPOAgent.load(agent_path) 
    # agent = RandomAgent.load(agent_path)

    result, _ = agent_environment_episode_loop(env, agent, args.test_episodes, training=True)
    
    if args.save_test:
        exp_dir = os.path.join(args.res_dir, f"test_{args.test_agent_path}_{seed}_{args.test_env_name}")
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
            pickle.dump(result, f)

    env.close()