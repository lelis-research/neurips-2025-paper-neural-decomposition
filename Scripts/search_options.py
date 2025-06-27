from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


import os
import torch
import itertools
import copy
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
from Scripts.learn_options import Option, seq_loss_fn


def search_options(args):
    assert args.tmp_opt == "DecOption", f"Wrong tmp_opt in search_options: {args.tmp_opt}"

    exp_dir = os.path.join(args.res_dir, args.option_exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    all_option_prototypes = []

    # Generate trajectory and sub-trajectories       
    if not os.path.exists(os.path.join(exp_dir, "trajectories.pt")):
        agent_info = []
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

            print("extracting trajectory for: ", env_agent["env_name"])
            traj = extract_trajectory(agent, env)
            print("Len traj extracted: ", len(traj))
            if len(traj) > 300:
                continue

            option_protos = []
            # for i in range(2, len(traj)):
            for i in [3,4]:
                option_proto = Option(agent.actor_critic.actor, 
                                      NetworkMasker(agent.actor_critic.actor, mask_type=args.mask_type).mask_logits,
                                      i)
                option_protos.append(option_proto)
            all_option_prototypes.append(option_protos)
            
            all_trajectories.append(traj)

            agent_info.append({
                "agent": agent,
                "env_name": env_agent["env_name"],
            })
        
        traj_data = {
            "agent_info": agent_info,
            "all_trajectories": all_trajectories,
        }
        torch.save(traj_data, os.path.join(exp_dir, "trajectories.pt"))
    else:
        print("\n\n", "*"*20, "LOADING TRAJECTORIES", "*"*20)
        traj_data = torch.load(os.path.join(exp_dir, "trajectories.pt"), weights_only=False)
        agent_info = traj_data["agent_info"]
        all_trajectories = traj_data["all_trajectories"]

        for traj, ag_info in zip(all_trajectories, agent_info):
            option_protos = []
            agent = ag_info['agent'] 
            # for i in range(2, len(traj)):
            for i in [3,4]:
                option_proto = Option(agent.actor_critic.actor, 
                                      NetworkMasker(agent.actor_critic.actor, mask_type=args.mask_type).mask_logits,
                                      i)
                option_protos.append(option_proto)
            all_option_prototypes.append(option_protos)

    for traj in all_trajectories:
        print("traj actions:", [pair[1] for pair in traj])
    

    file_name = f"selected_options_nolimit.pt" if args.max_num_options is None else f"selected_options_{args.max_num_options}.pt"
    if not os.path.exists(os.path.join(exp_dir, file_name)):
        print("\n\n", "*"*20, "SELECTING BEST OPTIONS", "*"*20)
        print(list(all_option_prototypes[0][0].masked_actor.mask_logits.values()))
        all_combos = []
        for option_lst in all_option_prototypes:
            for option in option_lst:
                mask_combos = {}
                for name, param in option.masked_actor.mask_logits.items():
                    size = param.shape[1]
                    values = [0,1,-1]
                    combinations = list(itertools.product(values, repeat=size))  # shape: [num_combinations, size]
                    onehot_map = {
                        -1: [1, 0, 0],
                        0: [0, 1, 0],
                        1: [0, 0, 1]
                    }
                    onehot_encoded = [
                        [onehot_map[v] for v in combo]  # flatten per row
                        for combo in combinations
                    ]

                    # Step 5: Convert to tensor
                    onehot_tensor = torch.tensor(onehot_encoded, dtype=torch.int).transpose(1,-1)
                    mask_combos[name] = onehot_tensor # COMB * 3 * SIZE
                    print(f"Combinations shape for {name}: {mask_combos[name].shape}")
                all_combos.append((option, mask_combos))

        loss_fn = partial(
                seq_loss_fn,
                all_trajectories,
                tol=args.action_dif_tolerance
            )
        
        if args.selection_type == "greedy":
            previous_loss = None
            best_loss = None
            best_options = []
            while (previous_loss is None or best_loss < previous_loss) and len(best_options) < 5:
                previous_loss = best_loss
                cur_best_loss = None
                cur_best_option = None
                # Trying each one of the combinations
                for option_proto, combos_dict in all_combos:
                    tasks = []
                    for mask_combo_combo in itertools.product(*list(combos_dict.values())): # NOTE: it should be and ordered list of values
                        # mask_combo_combo: [mask_combo for all params]
                        tasks.append((option_proto, mask_combo_combo, loss_fn, best_options))
                
                    with Pool(processes=args.num_workers) as pool:
                        for option, loss in tqdm(
                            pool.imap_unordered(_one_combo, tasks),
                            desc="Restarts",
                            unit="run"
                        ):
                            # track global best
                            if cur_best_loss is None or loss < cur_best_loss:
                                cur_best_loss = loss
                                cur_best_option = option
                                
                if best_loss is None or cur_best_loss < best_loss:
                    best_loss = cur_best_loss
                    best_options.append(cur_best_option)


        elif args.selection_type == "local_search":
            options_lst = []
            for option_proto, combos_dict in all_combos:
                for mask_combo_combo in itertools.product(*list(combos_dict.values())):                    
                    option_cpy = copy.deepcopy(option_proto)
                    option_cpy.update_mask(mask_combo_combo)
                    options_lst.append(option_cpy)
            best_options, best_loss = find_best_subset_stochastic_parallel(options_lst, 
                                                                           loss_fn=loss_fn, 
                                                                            max_iters=args.hc_iterations, 
                                                                            restarts=args.hc_restarts, 
                                                                            neighbor_samples=args.hc_neighbor_samples,
                                                                            max_size=args.max_num_options, 
                                                                            num_workers=args.num_worker)
        else:
            raise NotImplementedError    
        
        print("Best loss: ", best_loss, "Number of options: ", len(best_options))
        torch.save(best_options, os.path.join(exp_dir, file_name))
    else:
        print("\n\n", "*"*20, "LOADING SELECTED OPTIONS", "*"*20)
        best_options = torch.load(os.path.join(exp_dir, file_name), weights_only=False)
    
def _one_combo(args):
    option_proto, mask_combo, loss_fn, option_lst = args
    option_proto.update_mask(mask_combo)
    option = option_proto

    loss = loss_fn(option_lst + [option])
    return option, loss