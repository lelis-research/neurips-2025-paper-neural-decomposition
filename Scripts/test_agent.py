import numpy as np
import torch
import random
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

from Agents.PPOAgentOption import PPOAgentOption
from Agents.PPOAgent import PPOAgent
from Agents.RandomAgent import RandomAgent
from Agents.DQNAgent import DQNAgent
from Agents.NStepDQNAgent import NStepDQNAgent
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
                  render_mode=None,)
    
    agent_path = os.path.join(args.res_dir, args.test_agent_path, "best.pt")

    writer = None
    if args.save_test:
        exp_dir = os.path.join(args.res_dir, f"Test_{args.test_agent_path}_{seed}_{args.test_env_name}")        
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError(f"Experiment directory {exp_dir} already exists.")
        writer = SummaryWriter(log_dir=exp_dir)


    try:
        print("Tryig PPOAgent Loading")
        agent = PPOAgent.load(agent_path)
        # print("Loading DQN")
        # agent = DQNAgent.load(agent_path)
        # print("Loading NStep DQN")
        # agent = NStepDQNAgent.load(agent_path)
    except Exception as e1:
        print(f"PPOAgent loading failed with error: {e1}")
        try:
            print("Tryig PPOAgentOption Loading")
            agent = PPOAgentOption.load(agent_path)
        except Exception as e2:
            print(f"PPOAgentOption.load also failed with error: {e2}")
            print("Loading Both Agents Failed")
            exit(1)
    # agent = RandomAgent.load(agent_path)

    result, _ = agent_environment_episode_loop(env, agent, args.test_episodes, training=False,  
                                               writer=writer, save_frame_freq=1, greedy=True)
    
    success = sum([ep['episode_return'] > 0 for ep in result])
    
    print("*********************************************")
    print(f"number of succesful park from 100: {success}")
    # if args.save_test:
    #     exp_dir = os.path.join(args.res_dir, f"test_{args.test_agent_path}_{seed}_{args.test_env_name}")
    #     os.makedirs(exp_dir)
    #     with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
    #         pickle.dump(result, f)

    env.close()