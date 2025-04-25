import os
import torch
from Experiments.LearnOptions import extract_trajectory, generate_subtrajectories, learn_mask, find_best_subset, find_best_subset_stochastic
from Experiments.LevinLoss import levin_loss_continuous_with_maxlen
from Environments.GetEnvironment import get_env
from Agents.PPOAgent import PPOAgent

class option():
    def __init__(self, actor_mean, mask, max_len):
        self.actor_mean = actor_mean
        self.mask = mask
        self.max_len = max_len
    
    def act(self, observation):
        state = torch.from_numpy(observation).float()
        masked_state = state + self.mask
        action = self.actor_mean(masked_state)

        return action

def loss_fn(all_traj, options_lst):
    loss = 0
    for traj in all_traj:
        loss += levin_loss_continuous_with_maxlen(traj, options_lst) / len(all_traj)
    return loss

def train_options(args):
    data = []
    all_traj = []
    # get all sub-trajectories
    for env_agent in args.env_agent_list:
        env = get_env(env_name=env_agent["env_name"],
                    env_params=env_agent["env_params"],
                    wrapping_lst=env_agent["env_wrappers"],
                    wrapping_params=env_agent["env_wrapping_params"],
                    )
        agent_path = os.path.join(args.res_dir, env_agent["agent_path"], "final.pt")
        agent = PPOAgent.load(agent_path) 

        traj = extract_trajectory(agent, env)
        sub_trajs = generate_subtrajectories(traj, 45, 46)
        all_traj.append(traj)
        data.append({
            "agent": agent,
            "env": env,
            "sub_trajs": sub_trajs
        })

    # get a mask for each sub-trajectory
    options_lst = []
    for i, data_point1 in enumerate(data):
        agent = data_point1["agent"]
        for j, data_point2  in enumerate(data):
            print(i, j)
            if i == j:
                continue
            for sub_traj in data_point2["sub_trajs"]:
                mask = learn_mask(agent, sub_traj, num_epochs=0)
                options_lst.append(option(agent.actor_critic.actor_mean, mask, len(sub_traj)))

    print("Num Options: ", len(options_lst))

    best_options, best_loss = find_best_subset_stochastic(options_lst, lambda x: loss_fn(all_traj, x))
    print(best_loss, len(best_options))
    print([o.mask for o in best_options])