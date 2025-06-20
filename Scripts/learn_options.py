from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


import os
import torch
import pickle
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from torch.utils.tensorboard import SummaryWriter


from Experiments.LearnOptions import extract_trajectory, generate_subtrajectories, learn_mask, fine_tune_policy, find_best_subset_stochastic_parallel
from Experiments.LevinLoss import levin_loss_continuous_with_maxlen, levin_loss_discrete
from Environments.GetEnvironment import get_env
# from Agents.PPOAgent import PPOAgent
from Agents.PPOAgentOption import PPOAgentOption
from Agents.A2CAgentOption import A2CAgentOption
from Agents.A2CAgent import A2CAgent
from Experiments.EnvAgentLoops import agent_environment_step_loop, agent_environment_episode_loop
from Networks.MaskedNetwork import NetworkMasker
            
class Option():
    def __init__(self, actor, mask, max_len, info={}):
        self.actor = actor
        self.max_len = max_len
        self.info = info
        
        self.mask = mask
        if mask is not None:
            self.masked_actor = NetworkMasker(actor, mask)

    def update_mask(self, mask_combo):
        for i, (name, param) in enumerate(self.masked_actor.mask_logits.items()):
            tensor = torch.tensor(mask_combo[i]).reshape(param.shape)
            with torch.no_grad():  # avoid tracking in autograd
                self.masked_actor.mask_logits[name].copy_(tensor)
    
    def act(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        action = self.actor(state) if self.mask is None else self.masked_actor(state)
        
        # for discrete actions
        action = torch.argmax(action, dim=-1).item()
        return action

def seq_loss_fn(all_traj, options_lst, tol=1e-3):
    loss = 0
    for e, traj in enumerate(all_traj):
        # loss += levin_loss_continuous_with_maxlen(traj, options_lst, tol=tol) / len(all_traj)
        loss += levin_loss_discrete(traj, options_lst, num_actions=7) / len(all_traj)
    return loss

def parallel_loss_fn(all_traj, options_lst, num_workers=4, tol=1e-3):
    """
    Parallel computation of loss over all_traj using threads
    (so it can be called from inside a multiprocessing worker).
    """
    # worker_fn = partial(
    #     levin_loss_continuous_with_maxlen,
    #     options=options_lst,
    #     tol=tol
    # )
    worker_fn = partial(
        levin_loss_discrete,
        options=options_lst,
        num_actions=7,
    )
    n = len(all_traj)
    # use threads rather than processes
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        losses = list(pool.map(worker_fn, all_traj))
    return sum(losses) / n

def train_one_option(args):
    """
    Worker function to process one (i,j,sub_traj) task.
    Returns either an Option or None.
    """
    i, j, sub_traj1_env, sub_traj2_env, agent, sub_traj, baseline, mask_epochs, tol, mask_type = args
    # create a local tqdm just for the epochs if you really need it,
    # otherwise you can omit per-task bars.
    if baseline == "Mask":
        mask = learn_mask(agent, sub_traj,
                          num_epochs=mask_epochs,
                          tol=tol, mask_type=mask_type)
        if mask is None:
            return None
        info = {
            "org_policy_env": sub_traj1_env,
            "sub_traj_env": sub_traj2_env,
        }
        return Option(agent.actor_critic.actor,
                      mask,
                      len(sub_traj),
                      info=info)

    elif baseline == "FineTune":
        actor_critic = fine_tune_policy(agent, sub_traj,
                                        num_epochs=mask_epochs,
                                        tol=tol)
        if actor_critic is None:
            return None
        info = {
            "org_policy_env": sub_traj1_env,
            "sub_traj_env": sub_traj2_env,
        }
        return Option(actor_critic.actor,
                      None,
                      len(sub_traj),
                      info=info)

    elif baseline == "DecWhole":
        info = {
            "org_policy_env": sub_traj1_env,
            "sub_traj_env": sub_traj2_env,
        }
        return Option(agent.actor_critic.actor,
                      None,
                      len(sub_traj),
                      info=info)
    return None

def train_options(args):
    exp_dir = os.path.join(args.res_dir, args.option_exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    if args.baseline == "Transfer":
        # No training, just copy the base policies
        options_lst = []
        for env_agent in args.env_agent_list:
            agent_path = os.path.join(args.res_dir, env_agent["agent_path"], "final.pt")
            agent = A2CAgent.load(agent_path) 
            options_lst.append(Option(agent.actor_critic.actor, None, 1))
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
                            max_steps=env_agent["env_max_steps"],
                            )
                agent_path = os.path.join(args.res_dir, env_agent["agent_path"], "final.pt")
                agent = A2CAgent.load(agent_path) 

                print("extracting trajectory for: ", env_agent["env_name"], agent_path)
                traj = extract_trajectory(agent, env)
                print("Len traj extracted: ", len(traj))
                if len(traj) > 300:
                    continue
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

        for traj in all_trajectories:
            print("traj actions:", [pair[1] for pair in traj])
        
        # Train the masks or fine-tuning for options
        if not os.path.exists(os.path.join(exp_dir, "all_options.pt")):        
            print("\n\n", "*"*20, "TRAINING OPTIONS", "*"*20)
            # get an option for each sub-trajectory
            import copy
            
            tasks = []
            for i, sub_traj1 in enumerate(all_sub_trajectories):
                for j, sub_traj2 in enumerate(all_sub_trajectories):
                    if i == j:
                        continue
                    for sub in sub_traj2["sub_traj"]:
                        tasks.append((
                            i,
                            j,
                            sub_traj1["env_name"],
                            sub_traj2["env_name"],
                            sub_traj1["agent"],
                            copy.deepcopy(sub),
                            args.baseline,
                            args.mask_epochs,
                            args.action_dif_tolerance,
                            args.mask_type if args.baseline == "Mask" else None
                        ))

            options_lst = []
            # 2) Spawn a pool and process with a progress bar
            with Pool(processes=args.num_worker) as pool:
                try:
                    for opt in tqdm(pool.imap_unordered(train_one_option, tasks),
                                    total=len(tasks),
                                    desc="Building Options"):
                        if opt is not None:
                            options_lst.append(opt)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
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
            loss_fn = partial(
                seq_loss_fn,
                all_trajectories,
                tol=args.action_dif_tolerance
            )
            best_options, best_loss = find_best_subset_stochastic_parallel(options_lst, 
                                                                           loss_fn=loss_fn, 
                                                                            max_iters=args.hc_iterations, 
                                                                            restarts=args.hc_restarts, 
                                                                            neighbor_samples=args.hc_neighbor_samples,
                                                                            max_size=args.max_num_options, 
                                                                            num_workers=args.num_worker)
            
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
    if not os.path.exists(os.path.join(option_dir, file_name)):
        print("Selected Options Doesn't exists!")
        return None
    best_options = torch.load(os.path.join(option_dir, file_name), weights_only=False)

    print(f"Loaded Options from: {os.path.join(option_dir, file_name)}")
    print("Num options: ", len(best_options))
    
    test_option_dir = f"{option_dir}_{args.test_option_env_name}_{file_name[:-3]}_{args.option_name_tag}"
    env = get_env(env_name=args.test_option_env_name,
                  env_params=args.test_option_env_params,
                  wrapping_lst=args.test_option_env_wrappers,
                  wrapping_params=args.test_option_wrapping_params,
                  render_mode=args.test_option_render_mode,
                  max_steps=args.test_option_env_max_steps
                  )
    print(f"Obs Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # keys = ["gamma", "lamda",
    #             "epochs", "total_steps", "rollout_steps", "num_minibatches",
    #             "flag_anneal_step_size", "step_size",
    #             "entropy_coef", "critic_coef",  "clip_ratio", 
    #             "flag_clip_vloss", "flag_norm_adv", "max_grad_norm",
    #             "flag_anneal_var", "var_coef",
    #             ]
    keys = ["gamma", "step_size", "rollout_steps", "lamda"]
    
    agent_kwargs = {k: getattr(args, k) for k in keys}

    agent = A2CAgentOption(env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space, 
                           env.single_action_space if hasattr(env, "single_action_space") else env.action_space, 
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
