import os
import torch
import pickle
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from torch.utils.tensorboard import SummaryWriter

from Experiments.LearnOptions import extract_trajectory, generate_subtrajectories, learn_mask, find_best_subset, find_best_subset_stochastic, fine_tune_policy
from Experiments.LevinLoss import levin_loss_continuous_with_maxlen
from Environments.GetEnvironment import get_env
from Agents.PPOAgent import PPOAgent
from Agents.PPOAgentOption import PPOAgentOption
from Experiments.EnvAgentLoops import agent_environment_step_loop, agent_environment_episode_loop

class Option():
    def __init__(self, actor_mean, mask, max_len, info={}):
        self.actor_mean = actor_mean
        self.mask = mask
        self.max_len = max_len
        self.info = info
    
    def act(self, observation):
        state = torch.from_numpy(observation).float()
        masked_state = state + self.mask
        action = self.actor_mean(masked_state)

        return action

def loss_fn(all_traj, options_lst, tol=1e-3):
    loss = 0
    for e, traj in enumerate(all_traj):
        loss += levin_loss_continuous_with_maxlen(traj, options_lst, tol=tol) / len(all_traj)
    return loss

def parallel_loss_fn(all_traj, options_lst, num_workers=4, tol=1e-3):
    """
    Parallel computation of loss over all_traj using a picklable partial.
    """
    # bake in options_lst and tol
    worker_fn = partial(
        levin_loss_continuous_with_maxlen,
        options=options_lst,
        tol=tol
    )
    n = len(all_traj)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        # map the single-argument worker_fn over all_traj
        losses = list(pool.map(worker_fn, all_traj))
    return sum(losses) / n

def train_options(args):
    exp_dir = os.path.join(args.res_dir, args.option_exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    if args.baseline == "transfer":
        # No training, just copy the base policies
        options_lst = []
        for env_agent in args.env_agent_list:
            agent_path = os.path.join(args.res_dir, env_agent["agent_path"], "final.pt")
            agent = PPOAgent.load(agent_path) 
            options_lst.append(Option(agent.actor_critic.actor_mean, 0, 1))
        best_options = options_lst
        torch.save(best_options, os.path.join(exp_dir, "selected_options.pt"))
    else:
        # Generate trajectory and sub-trajectories       
        if not os.path.exists(os.path.join(exp_dir, "trajectories.pt")):
            all_sub_trajectories = []
            all_trajectories = []
            print("\n\n", "*"*20, "GETTING TRAJECTORIES", "*"*20)
            # get all sub-trajectories
            for env_agent in args.env_agent_list:
                env = get_env(env_name=env_agent["env_name"],
                            env_params=env_agent["env_params"],
                            wrapping_lst=env_agent["env_wrappers"],
                            wrapping_params=env_agent["env_wrapping_params"],
                            )
                agent_path = os.path.join(args.res_dir, env_agent["agent_path"], "final.pt")
                agent = PPOAgent.load(agent_path) 

                print("extracting trajectory for: ", env_agent["env_name"])
                traj = extract_trajectory(agent, env)
                print("Len traj extracted: ", len(traj))
                sub_traj = generate_subtrajectories(traj, args.sub_trajectory_min_len, args.sub_trajectory_max_len)
                print("Num sub-traj extracted: ", len(sub_traj))
                all_trajectories.append(traj)

                all_sub_trajectories.append({
                    "agent": agent,
                    "sub_traj": sub_traj,
                    "env_name": env_agent["env_name"],
                })
            
            traj_data = {
                "all_sub_trajectories": all_sub_trajectories,
                "all_trajectories": all_trajectories,
            }
            torch.save(traj_data, os.path.join(exp_dir, "trajectories.pt"))
        else:
            print("\n\n", "*"*20, "LOADING TRAJECTORIES", "*"*20)
            traj_data = torch.load(os.path.join(exp_dir, "trajectories.pt"), weights_only=False)
            all_sub_trajectories = traj_data["all_sub_trajectories"]
            all_trajectories = traj_data["all_trajectories"]
            
        print("\n","*** Total num sub-trajectories: ", sum([len(d["sub_traj"]) for d in all_sub_trajectories]), " ***")
        
        # Train the masks or fine-tuning for options
        if not os.path.exists(os.path.join(exp_dir, "all_options.pt")):        
            print("\n\n", "*"*20, "TRAINING MASKS", "*"*20)
            # get a mask for each sub-trajectory

            options_lst = []
            
            # count total masks (i.e. total sub_trajs across all agent/traj pairs, excluding i==j)
            total_masks = (len(all_sub_trajectories) - 1) * sum(len(d["sub_traj"]) for d in all_sub_trajectories)
            mask_pbar = tqdm(total=total_masks, desc="Masks", position=0)
            for i, sub_traj1 in enumerate(all_sub_trajectories):
                agent = sub_traj1["agent"]
                for j, sub_traj2  in enumerate(all_sub_trajectories):
                    if i == j:
                        continue
                    for sub_traj in sub_traj2["sub_traj"]:
                        epoch_pbar = tqdm(
                            total=args.mask_epochs,
                            desc=f"Agent {i}, Traj {j}",
                            leave=False,
                            position=1,
                        )
                        
                        if args.baseline == "mask":
                            mask = learn_mask(agent, sub_traj, num_epochs=args.mask_epochs, pbar=epoch_pbar, tol=args.action_dif_tolerance)
                            epoch_pbar.close()
                            if mask is not None: # if the mask is None, the learning wasn't successful
                                info = {
                                    "org_policy_env": sub_traj1["env_name"],
                                    "sub_traj_env": sub_traj2["env_name"],
                                }
                                options_lst.append(Option(agent.actor_critic.actor_mean, mask, len(sub_traj), info=info))
                        
                        elif args.baseline == "tune":
                            actor_critic = fine_tune_policy(agent, sub_traj, num_epochs=args.mask_epochs, pbar=epoch_pbar, tol=args.action_dif_tolerance)
                            epoch_pbar.close()
                            if actor_critic is not None: # if the mask is None, the learning wasn't successful
                                info = {
                                    "org_policy_env": sub_traj1["env_name"],
                                    "sub_traj_env": sub_traj2["env_name"],
                                }
                                options_lst.append(Option(actor_critic.actor_mean, 0, len(sub_traj), info=info))
                        
                        elif args.baseline == "decwhole":
                            epoch_pbar.close()
                            info = {
                                "org_policy_env": sub_traj1["env_name"],
                                "sub_traj_env": sub_traj2["env_name"],
                            }
                            options_lst.append(Option(agent.actor_critic.actor_mean, 0, len(sub_traj), info=info))
                            
                        mask_pbar.update(1)
            mask_pbar.close()      
            torch.save(options_lst, os.path.join(exp_dir, "all_options.pt"))
        else:
            print("\n\n", "*"*20, "LOADING MASKS", "*"*20)
            options_lst = torch.load(os.path.join(exp_dir, "all_options.pt"), weights_only=False)
        
        for option in options_lst:
            print(option.info, option.max_len)
        print("\n", "*** Total number of options: ", len(options_lst), " ***")
        
        # Select the best combination of options to reduce Levin loss
        file_name = f"selected_options_nolimit.pt" if args.max_num_options is None else f"selected_options_{args.max_num_options}.pt"
        if not os.path.exists(os.path.join(exp_dir, file_name)):
            print("\n\n", "*"*20, "SELECTING BEST OPTIONS", "*"*20)
            best_options, best_loss = find_best_subset_stochastic(options_lst, lambda x: parallel_loss_fn(all_trajectories, x, tol=args.action_dif_tolerance), 
                                                                max_iters=args.hc_iterations, restarts=args.hc_restarts, neighbor_samples=args.hc_neighbor_samples,
                                                                max_size=args.max_num_options)
            
            print("Best loss: ", best_loss)
            torch.save(best_options, os.path.join(exp_dir, file_name))
        else:
            print("\n\n", "*"*20, "LOADING SELECTED OPTIONS", "*"*20)
            best_options = torch.load(os.path.join(exp_dir, file_name), weights_only=False)

    print("\n", "*** Num selected options: ", len(best_options), " ***")
    for option in best_options:
        print(option.info, option.max_len)

def test_options(args):
    option_dir = os.path.join(args.res_dir, args.option_exp_name)
    file_name = f"selected_options.pt" if args.max_num_options is None else f"selected_options_{args.max_num_options}.pt"
    best_options = torch.load(os.path.join(option_dir, file_name), weights_only=False)
    print(f"Loaded Options from: {os.path.join(option_dir, file_name)}")
    print("Num options: ", len(best_options))
    
    test_option_dir = f"{option_dir}_{args.test_option_env_name}_{file_name}"

    env = get_env(env_name=args.test_option_env_name,
                  env_params=args.test_option_env_params,
                  wrapping_lst=args.test_option_env_wrappers,
                  wrapping_params=args.test_option_wrapping_params,
                  render_mode=args.test_option_render_mode,
                  max_steps=args.test_option_env_max_steps
                  )
    print(f"Obs Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    ppo_keys = ["gamma", "lamda",
                "epochs", "total_steps", "rollout_steps", "num_minibatches",
                "flag_anneal_step_size", "step_size",
                "entropy_coef", "critic_coef",  "clip_ratio", 
                "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
                "flag_anneal_var", "var_coef",
                ]
    agent_kwargs = {k: getattr(args, k) for k in ppo_keys}

    agent = PPOAgentOption(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                            best_options,
                            device=args.device,
                            **agent_kwargs
                            )
     
    writer = None
    if args.option_save_results:
        if not os.path.exists(test_option_dir):
            os.makedirs(test_option_dir)
        else:
            raise ValueError(f"Experiment directory {test_option_dir} already exists.")
        writer = SummaryWriter(log_dir=test_option_dir)

    if args.exp_options_total_steps > 0 and args.exp_options_total_episodes == 0:
        result, best_agent = agent_environment_step_loop(env, agent, args.exp_options_total_steps, writer=writer, save_frame_freq=args.option_save_frame_freq)
    elif args.exp_options_total_episodes > 0 and args.exp_options_total_steps == 0:
        result, best_agent = agent_environment_episode_loop(env, agent, args.exp_options_total_episodes, writer=writer, save_frame_freq=args.option_save_frame_freq)
    else:
        raise ValueError("Both steps and episodes are greater than 0")
    

    if args.save_results:
        with open(os.path.join(test_option_dir, "res.pkl"), "wb") as f:
            pickle.dump(result, f)
        agent.save(os.path.join(test_option_dir, "final.pt"))
        best_agent.save(os.path.join(test_option_dir, "best.pt"))
    env.close()
    return result
