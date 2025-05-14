import copy
from dataclasses import dataclass
import os
import pickle
import random
import traceback

from ordered_set import OrderedSet
from environments.environments_combogrid import DIRECTIONS, PROBLEM_NAMES as COMBO_PROBLEM_NAMES
from environments.environments_combogrid_gym import ComboGym
from environments.environments_minigrid import get_training_tasks_simplecross
import torch
import numpy as np
import math
import tyro

import concurrent
from typing import List, Tuple, Union
from agents.policy_guided_agent import PPOAgent
from pipelines.option_discovery import get_single_environment, save_options, regenerate_trajectories
from utils import utils
from utils.utils import timing_decorator


@dataclass
class Args:
    # exp_name: str = "extract_learnOption_noReg_excludeGoal"
    # exp_name: str = "extract_learnOption_regularization"
    # exp_name: str = "debug"
    exp_name: str = "extract_fineTuning_notFiltered"
    # exp_name: str = "extract_fineTuning"
    # exp_name: str = "extract_decOptionWhole_sparseInit"
    # exp_name: str = "extract_learnOptions_randomInit_discreteMasks"
    # exp_name: str = "extract_learnOptions_randomInit_pitisFunction"
    """the name of this experiment"""
    env_seeds: Union[List, str, Tuple] = (0,1,2,3)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""
    # model_paths: List[str] = (
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    # model_paths: List[str] = (
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_lr0.001_clip0.2_ent0.1_envsd2'
    # )
    # model_paths: List[str] = (
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.0005_clip0.25_ent0.1_envsd0',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd1',
    #     'train_ppoAgent_sparseInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_lr0.001_clip0.2_ent0.1_envsd2',
    #     )
    model_paths: List[str] = (
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR',
    )

    # These attributes will be filled in the runtime
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    problems: List[str] = ()
    """the name of the problems the agents were trained on; To be filled in runtime"""

    # Algorithm specific arguments
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    choices from [ComboGrid, MiniGrid-SimpleCrossingS9N1-v0]
    """
    cpus: int = 4
    """"The number of CPUTs used in this experiment."""
    
    # hyperparameters
    game_width: int = 5
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""
    number_actions: int = 3

    # learning
    fine_tuning_steps: int = 2_000
    """"""
    input_update_frequency: int = 1
    """"""
    selection_type: str = "local_search"
    """It's either `local_search` or `greedy`"""
    cache_path: str = ""
    """Path to the directory where the options are saved"""
    reg_coef: float = 0.0
    # reg_coef: float = 110.03
    filtering_inapplicable: bool = False
    # max_num_options: int = 10
    max_num_options: int = 5

    # Script arguments
    seed: int = 0
    """The seed used for reproducibilty of the script"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BASELINE0_Combogrid"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    log_path: str = "outputs/logs/"
    """The name of the log file"""
    log_level: str = "INFO"
    """The logging level"""

def process_args() -> Args:
    args: Args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}' + \
        f'_envsd{",".join(map(str, args.env_seeds))}'
        if 'selection_type' in vars(args):
            args.exp_id += f'_selectType{args.selection_type}'
        args.exp_id += f'_reg{args.reg_coef}' # TODO: not conditioned correctly
        args.exp_id += f'maxNumOptions{args.max_num_options}'

    # updating log path
    args.log_path = os.path.join(args.log_path, args.exp_id, f"seed={str(args.seed)}")
    
    # Processing seeds from commands
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        start, end = map(int, args.env_seeds.split(","))
        args.env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    if args.env_id == "ComboGrid":
        args.problems = [COMBO_PROBLEM_NAMES[seed] for seed in args.env_seeds]
    elif args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        args.problems = [args.env_id + f"_{seed}" for seed in args.env_seeds]
        
    return args


class FineTuning:
    def __init__(self, args: Args, logger):
        self.args = args
        self.logger = logger
        self.selection_type = args.selection_type
        self.number_actions = args.number_actions

        self.selection_type = args.selection_type
        if args.cache_path == "":
            args.cache_path = "binary/cache/"
            self.option_candidates_path = os.path.join(args.cache_path, args.exp_id, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, args.exp_id, f"seed={args.seed}", "option_cache.pkl")
        else:
            self.option_candidates_path = os.path.join(args.cache_path, f"seed={args.seed}", "data.pkl")
            self.option_cache_path = os.path.join(args.cache_path, f"seed={args.seed}", "option_cache.pkl")

        print(f"Option candidates path: {self.option_candidates_path}")
        print(f"exists:{os.path.exists(self.option_candidates_path)}")

        self.levin_loss = LevinLossActorCritic(self.logger, alpha=args.reg_coef)
        
        
    def _train_last_layer(self, trajectories, target_problem, s, length, primary_agent: PPOAgent):
        """
        Train the last layer of the agent using the given trajectory segment.
        """
        # Extract the trajectory segment
        trajectory = trajectories[target_problem].slice(s, n=length)
        agent = copy.deepcopy(primary_agent)
        last_layer = list(agent.actor)[-1]
        optimizer = torch.optim.Adam(last_layer.parameters(), lr=1e-3)

        init_loss = None

        steps = 0
        while steps < self.args.fine_tuning_steps:
            # Iterate over trajectories
            if steps >= self.args.fine_tuning_steps:
                break

            envs = trajectory.get_state_sequence()
            actions = trajectory.get_action_sequence()
            agent.eval()

            new_trajectory = agent.run_fixed_prefix(envs, trajectory.get_length())

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions))
            if not init_loss:
                init_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress
            steps += 1

        trajectory = list(trajectories.values())[0]
        envs = trajectory.get_state_sequence()
        new_trajectory = agent.run_fixed_prefix(envs, trajectory.get_length())

        applicable = new_trajectory.get_action_sequence() == trajectory.get_action_sequence()

        return agent, init_loss, loss.item(), applicable

    @timing_decorator
    def discover(self):
        """
        Discover the best option fine-tuning the last layer of the actor network.
        """
        trajectories = regenerate_trajectories(self.args, verbose=True, logger=self.logger)

        if not os.path.exists(self.option_candidates_path):

            option_candidates = []

            for target_seed, target_problem in zip(self.args.env_seeds, self.args.problems):
                self.logger.info(f'Extracting options using trajectory segments from {target_problem}, env_seed={target_seed}')
                mimicing_agents = {}

                t_length = trajectories[target_problem].get_length()

                for primary_seed, primary_problem, primary_model_directory in zip(self.args.env_seeds, self.args.problems, self.args.model_paths):
                    if primary_problem == target_problem:
                        continue
                    model_path = f'binary/models/{primary_model_directory}/seed={self.args.seed}/ppo_first_MODEL.pt'
                    primary_env = get_single_environment(self.args, seed=primary_seed)
                    primary_agent = PPOAgent(primary_env, hidden_size=self.args.hidden_size)
                    primary_agent.load_state_dict(torch.load(model_path))
                    mimicing_agents[primary_problem] = (primary_seed, model_path, primary_agent)

                print(f"Number of mimicing agents: {len(mimicing_agents)}")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                    # Submit tasks to the executor with all required arguments
                    futures = []
                    for length in range(2, t_length + 1):
                        # if length != 3:
                        #     continue
                        for s in range(t_length - length + 1):
                            # actions = trajectories[target_problem].slice(s, n=length).get_action_sequence()
                            # if actions not in [[0,0,1], [2,1,0],[1,0,2], [0,1,2]]:
                            #     continue
                            for primary_problem, (primary_seed, primary_model_path, primary_agent) in mimicing_agents.items():
                                future = executor.submit(
                                    self._train_last_layer, trajectories, target_problem, s, length, primary_agent)
                                future.s = s,
                                future.length = length
                                future.primary_problem = primary_problem
                                future.primary_env_seed = primary_seed
                                future.primary_model_path = primary_model_path
                                futures.append(future)

                    # Process the results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            agent, init_loss, final_loss, applicable = future.result()
                            if (not self.args.filtering_inapplicable) or applicable:
                                option_candidates.append((agent.state_dict(),
                                                        future.primary_problem, 
                                                    target_problem, 
                                                    future.primary_env_seed, 
                                                    target_seed, 
                                                    future.length, 
                                                    future.primary_model_path, 
                                                    future.s))
                                self.logger.info(f'Progress: segment:{future.s} of length {future.length}, primary_problem={future.primary_problem} done. init_loss={init_loss}, final_loss={final_loss}')
                                actions = []
                                for i in range(future.length):
                                    t_env = trajectories[target_problem].get_trajectory()[future.s[0] + i][0]
                                    o = torch.tensor(t_env.get_observation(), dtype=torch.float32)
                                    actions.append(agent.get_action_and_value(o, deterministic=True)[0].item())
                                t_actions = trajectories[target_problem].get_action_sequence()[future.s[0]: future.s[0]+ future.length]
                                assert actions == t_actions, f"Agent {agent.extra_info} failed to mimic the action {t_actions} in state {t_env} with action {actions} at setgment {future.s} of length {future.length}"
                        except Exception as exc:
                            self.logger.error(f'Segment:{future.s} of length {future.length} with primary_problem={future.primary_problem} generated an exception: {exc}')
                            traceback.print_exc()
                            return
                utils.logger_flush(self.logger)
            self.logger.debug("\n")
            self.logger.info("Saving parameters ... ")
            os.makedirs(os.path.dirname(self.option_candidates_path))
            with open(self.option_candidates_path, 'wb') as f:
                pickle.dump({'option_candidates': option_candidates, 'trajectories': trajectories}, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.info("Loading parameters ... ")
            with open(self.option_candidates_path, 'rb') as f:
                data = pickle.load(f)
                option_candidates = data['option_candidates']
                trajectories = data['trajectories']
        
        # ## DEBUGGING
        # all_options = []
        # for state_dict, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
        #     self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
        #     env = get_single_environment(self.args, seed=primary_env_seed)
        #     agent = PPOAgent(env, hidden_size=self.args.hidden_size)
        #     agent.load_state_dict(state_dict)
        #     agent.to_option(None, option_size, target_problem)
        #     agent.extra_info['primary_problem'] = primary_problem
        #     agent.extra_info['primary_env_seed'] = primary_env_seed
        #     agent.extra_info['target_problem'] = target_problem
        #     agent.extra_info['target_env_seed'] = target_env_seed
        #     agent.extra_info['segment'] = segment
        #     all_options.append(agent)

        # self.levin_loss.print_output_subpolicy_trajectory(all_options, trajectories, logger=self.logger)    
        # return 
        # ####


        if self.selection_type == "greedy":
            selected_options = self.select_greedy(option_candidates, trajectories)
        elif self.selection_type == "local_search":
            selected_options = self.select_by_local_search(option_candidates, trajectories)
        else:
            raise ValueError(f"Invalid selection type: {self.selection_type}")

        # printing selected options
        self.logger.info("Selected options:")
        for i in range(len(selected_options)):
            self.logger.info(f"Option #{i}:\n" + 
                        f"mask={selected_options[i].mask}\n" +
                        f"size={selected_options[i].option_size}\n" +
                        f"extra_info={selected_options[i].extra_info}\n" )

        save_options(options=selected_options, 
                    trajectories=trajectories,
                    args=self.args, 
                    logger=self.logger)

        utils.logger_flush(self.logger)

        self.levin_loss.print_output_subpolicy_trajectory(selected_options, trajectories, logger=self.logger)
        utils.logger_flush(self.logger)

        self.logger.info("Testing on each grid cell")
        for seed, problem in zip(self.args.env_seeds, self.args.problems):
            self.logger.info(f"Testing on each cell..., {problem}")
            self.levin_loss.evaluate_on_each_cell(options=selected_options, 
                                    trajectories=trajectories,
                                    problem_test=problem, 
                                    args=self.args, 
                                    seed=seed, 
                                    logger=self.logger)

        utils.logger_flush(self.logger)

    def select_greedy(self, option_candidates, trajectories):
        selected_options = []

        previous_loss = None
        best_loss = None

        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
                
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list

        while previous_loss is None or best_loss < previous_loss:
            previous_loss = best_loss

            best_loss = None
            best_option = None

            for primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
                self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
                env = get_single_environment(self.args, seed=primary_env_seed)
                agent = PPOAgent(env, hidden_size=self.args.hidden_size)
                agent.load_state_dict(torch.load(model_path))
                agent.to_option(None, option_size, target_problem)
                agent.extra_info['primary_problem'] = primary_problem
                agent.extra_info['primary_env_seed'] = primary_env_seed
                agent.extra_info['target_problem'] = target_problem
                agent.extra_info['target_env_seed'] = target_env_seed
                agent.extra_info['segment'] = segment

                loss_value = self.levin_loss.compute_loss_cached(options=selected_options + [agent],
                                                                 trajectory=chained_trajectory,
                                                                    joint_problem_name_list=joint_problem_name_list,
                                                                    problem_str="", 
                                                                    number_actions=self.number_actions
                )

                if best_loss is None or loss_value < best_loss:
                    best_loss = loss_value
                    best_option = agent

            selected_options.append(best_option)

            if previous_loss is None or best_loss < previous_loss:
                self.logger.info(f"Added option #{len(selected_options)}; Levin loss of the current selected set: {best_loss} on all trajectories")
            utils.logger_flush(self.logger)

        # remove the last automaton added
        num_options = len(selected_options)
        selected_options = selected_options[:num_options - 1]
        return selected_options

    def _compute_sample_weight(self, option_refs, possible_sequences, all_options):
        transitions = [0.1 for _ in range(len(option_refs))]
        for i, o_idx in enumerate(option_refs):
            option = all_options[o_idx]
            option_id = option.get_option_id()
            for problem_name, sub_cache in self.levin_loss.cache[option_id].items():
                for segment, (applicable, _) in sub_cache.items():
                    if applicable == True and (segment, segment + option.option_size) in possible_sequences[problem_name]:
                        transitions[i] += 1

        weights = np.array(transitions)
        weights /= weights.sum()
        return weights

    def _search_options_subset(self, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, worker_id):
        max_num_neighbours = 125
        
        random_generator = np.random.default_rng([worker_id, self.args.seed])
        all_option_refs = set(range(len(all_options)))
        
        max_num_options = min(max_num_options, len(all_option_refs))
        # length_weights = np.array([1/(i+2) for i in range(max_num_options + 1)])
        length_weights = np.array([1/(i*3+2) for i in range(max_num_options + 1)])
        length_weights /= np.sum(length_weights)
        subset_length = random_generator.choice(range(max_num_options + 1), p=length_weights)
        
        weights = self._compute_sample_weight(all_option_refs, all_possible_sequences, all_options)
        selected_options = set(random_generator.choice(list(all_option_refs), p=weights, size=subset_length, replace=False).tolist())
        possible_sequences = copy.deepcopy(all_possible_sequences)
        
        previous_cost = float('Inf')
        total_loss_calculations = 1
        steps = 0
        best_cost = 0
        for problem, trajectory in trajectories.items():
            cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in selected_options], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
            possible_sequences[problem] = possible_sequences[problem] - used_sequences
            best_cost += cost
        
        while (best_cost < previous_cost or steps == 0) and steps < max_steps:
            previous_cost = best_cost
            not_selected_options = all_option_refs - selected_options
            weights = self._compute_sample_weight(not_selected_options, possible_sequences, all_options)
            num_neighbours = min(max_num_neighbours, len(not_selected_options))
            sample_options = random_generator.choice(list(not_selected_options), p=weights, size=num_neighbours, replace=False).tolist()
            # assert len(sample_options) == (len(all_options) - len(selected_options)) == len(not_selected_options), f"Sampled options {len(sample_options)} should be equal to all options {len(all_options)} len of selected ones: {len(selected_options)}, len of not selected ones: {len(not_selected_options)}"

            neighbours = []
            for option in sample_options:
                assert option not in selected_options, f"Option {option.get_option_id()} should not be in selected options."
                if option not in selected_options:
                    if len(selected_options) < max_num_options:
                        neighbour = selected_options | {option}
                        neighbours.append(neighbour)
                    for option2 in selected_options:
                        neighbour = selected_options - {option2} | {option}
                        neighbours.append(neighbour)
            for option in selected_options:
                neighbour = selected_options - {option}
                neighbours.append(neighbour)
                

            # self.logger.info(f"Number of neighbours: {len(neighbours)}")
            i = 0
            for neighbour in neighbours:
                i += 1
                cost = 0
                remaining_sequences = copy.deepcopy(all_possible_sequences)
                for problem, trajectory in trajectories.items():
                    returned_cost, used_sequences = self.levin_loss.compute_loss_cached([all_options[idx] for idx in neighbour], 
                                                                                        trajectory, 
                                                                                        problem_str=problem, 
                                                                                        number_actions=self.number_actions)
                    remaining_sequences[problem] = remaining_sequences[problem] - used_sequences
                    cost += returned_cost
                    total_loss_calculations += 1
                if cost < best_cost:
                    selected_options = neighbour
                    possible_sequences = remaining_sequences
                    best_cost = cost
            steps += 1
        
        return best_cost, selected_options, {"total_loss_calculations": total_loss_calculations, "steps":steps}

    def _compute_option_applicability(self, option, trajectories):
        result = {problem: {} for problem in trajectories.keys()}
        applicable_count = 0
        for problem, trajectory in trajectories.items():
            t_len = trajectory.get_length()
            t = trajectory.get_trajectory()
            for s in range(t_len):
                actions = self.levin_loss._run(copy.deepcopy(t[s][0]), option, option.option_size)
                is_applicable = (len(actions) == option.option_size) and self.levin_loss.is_applicable(t, actions, s)
                if is_applicable:
                    applicable_count += 1
                result[problem][s] = (is_applicable, actions)
        return result, applicable_count

    def select_by_local_search(self, option_candidates, trajectories):

        all_options = []

        for state_dict, primary_problem, target_problem, primary_env_seed, target_env_seed, option_size, model_path, segment in option_candidates:
            self.logger.info(f'Evaluating the option trained on the segment {({segment[0]},{segment[0]+option_size})} from problem={target_problem}, env_seed={target_env_seed}, primary_problem={primary_problem}')
            env = get_single_environment(self.args, seed=primary_env_seed)
            agent = PPOAgent(env, hidden_size=self.args.hidden_size)
            agent.load_state_dict(state_dict)
            agent.to_option(None, option_size, target_problem)
            agent.extra_info['primary_problem'] = primary_problem
            agent.extra_info['primary_env_seed'] = primary_env_seed
            agent.extra_info['target_problem'] = target_problem
            agent.extra_info['target_env_seed'] = target_env_seed
            agent.extra_info['segment'] = segment
            all_options.append(agent)

        self.logger.info(f"Number of option_candidates: {len(all_options)}")

        if os.path.exists(self.option_cache_path):
            self.logger.info(f"Loading option cache from {self.option_cache_path}")
            with open(self.option_cache_path, 'rb') as f:
                self.levin_loss.cache = pickle.load(f)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
                # Submit tasks to the executor with all required arguments
                futures = set()
                for i, option in enumerate(all_options):
                    future = executor.submit(
                        self._compute_option_applicability, option, trajectories)
                    future.option_id = option.get_option_id()
                    futures.add(future)
                self.logger.info(f"Logging Checkpoint 1.")

                # Process the results as they complete
                total_applicable_count = 0
                progress = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        option_id = future.option_id
                        self.levin_loss.cache[option_id], n_applicable = future.result()
                        total_applicable_count += n_applicable
                        progress += 1
                        if progress % 100 == 0:
                            self.logger.info(f"Cache size: {len(self.levin_loss.cache)}, applicable_count: {total_applicable_count}")
                    except Exception as exc:
                        self.logger.error(f'Exception: {exc}')
                        traceback.print_exc()
                        return
            self.logger.info(f"Saving option cache to {self.option_cache_path}")
            with open(self.option_cache_path, 'wb') as f:
                pickle.dump(self.levin_loss.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Cache created. size: {len(self.levin_loss.cache)}")

        all_possible_sequences = {}
        for problem_name, trajectory in trajectories.items():
            trajectory = trajectory.get_trajectory()
            all_possible_sequences[problem_name] = set()
            t_length = len(trajectory)
            for length in range(2, t_length + 1):
                for s in range(0, t_length - length + 1):
                    all_possible_sequences[problem_name].add((s, s+length))

        chained_trajectory = None
        joint_problem_name_list = []
        for problem, trajectory in trajectories.items():

            if chained_trajectory is None:
                chained_trajectory = copy.deepcopy(trajectory)
            else:
                chained_trajectory.concat(trajectory)
            name_list = [problem for _ in range(len(trajectory._sequence))]
            joint_problem_name_list = joint_problem_name_list + name_list

        restarts = 200
        max_steps = 500
        max_num_options = self.args.max_num_options
        best_selected_options = []
        best_levin_loss_total = float('Inf')
        completed = 0

        for problem, trajectory in trajectories.items():
            cost, _ = self.levin_loss.compute_loss_cached([], 
                                                            trajectory, 
                                                            problem_str=problem, 
                                                            number_actions=self.number_actions)
            print(f"Cost of empty set: {cost} for problem {problem}")
            

        # for i in range(10):
        #     best_cost, selected_options, info = self._search_options_subset(max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
        #     total_loss_calculations = info['total_loss_calculations']
        #     steps = info['steps']
        #     # self.levin_loss.cache.update(cache)
        #     if best_cost < best_levin_loss_total:
        #         best_levin_loss_total = best_cost
        #         best_selected_options = selected_options
        #     completed += 1
        #     # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
        #     self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
        #     utils.logger_flush(self.logger)        


        with concurrent.futures.ProcessPoolExecutor(max_workers=self.args.cpus) as executor:
            # Submit tasks to the executor with all required arguments
            futures = set()
            for i in range(restarts):
                future = executor.submit(
                    self._search_options_subset, max_num_options, all_options, all_possible_sequences, trajectories, max_steps, i)
                self.logger.info(f"Restart {i} of {restarts} submitted.")
                futures.add(future)
            self.logger.info(f"Logging Checkpoint 2.")

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    best_cost, selected_options, info = future.result()
                    total_loss_calculations = info['total_loss_calculations']
                    steps = info['steps']
                    # self.levin_loss.cache.update(cache)
                    if best_cost < best_levin_loss_total:
                        best_levin_loss_total = best_cost
                        best_selected_options = selected_options
                    completed += 1
                    # self.logger.info(f"cache size: {len(self.levin_loss.cache)}")
                    self.logger.info(f"Restart {completed} of {restarts} complete. Selected Options: {selected_options}, total_loss_calculations={total_loss_calculations}, steps={steps}, Levin loss: {best_cost}, Best: {best_levin_loss_total}")
                    utils.logger_flush(self.logger)
                except Exception as exc:
                    self.logger.error(f'Exception: {exc}')
                    traceback.print_exc()
                    return

        best_selected_options = [all_options[idx] for idx in best_selected_options]
        self.levin_loss.remove_cache()
        
        # Removing redundant options
        def get_levin_loss(options, trajectories):
            cost = 0
            for problem, trajectory in trajectories.items():
                cost += self.levin_loss.compute_loss_cached(options, 
                                                trajectory, 
                                                problem_str=problem, 
                                                number_actions=3,
                                                cache_enabled=False)[0]
            return cost
    
        best_levin_loss = get_levin_loss(best_selected_options, trajectories)
        
        self.logger.info(f"Levin loss: {best_levin_loss}")
        options = copy.deepcopy(best_selected_options)
        while True:
            done = True
            best_loss_so_far = best_levin_loss
            for i in range(len(options)):
                options_cpy = copy.deepcopy(options)
                options_cpy = options_cpy[:i] + options_cpy[i+1:]
                levin_loss = get_levin_loss(options_cpy, trajectories)
                if levin_loss < best_loss_so_far:
                    best_loss_so_far = levin_loss
                    best_options_so_far = options_cpy
                    redundant_idx = i
                    done = False
            if not done:
                best_levin_loss = best_loss_so_far
                options = best_options_so_far
                self.logger.info(f"Levin loss without option #{redundant_idx}: {best_levin_loss}")
            else:
                break

        return list(options)


class LevinLossActorCritic:
    def __init__(self, logger, alpha=0):
        self.logger = logger
        self.alpha = alpha
        self.cache = {}

    def remove_cache(self):
        """
        This function removes the cache of the Levin loss. 
        """
        self.logger.info(f"Cache size: {len(self.cache)}")
        self.cache = {}
        self.logger.info("Cache removed.")

    def is_applicable(self, trajectory, actions, start_index):
        """
        This function checks whether an MLP is applicable in a given state. 

        An actor-critic agent is applicable if the sequence of actions it produces matches
        the sequence of actions in the trajectory. Note that we do not consider an
        actor-critic agent if it has less than 2 actions, as it would be equivalent to a 
        primitive action. 
        """
        if len(actions) <= 1 or len(actions) + start_index > len(trajectory):
            return False
        
        for i in range(len(actions)):
            if actions[i] != trajectory[i + start_index][1]:
                return False
        return True

    def _run(self, env: ComboGym, agent: PPOAgent, numbers_steps: int):
        """
        This function executes an option.

        It runs the option for the specified number of steps and it returns the actions taken for those steps. 
        """
        trajectory = agent.run_fixed_prefix(env, numbers_steps)
        actions = []
        for _, action in trajectory.get_trajectory():
            actions.append(action)

        assert len(actions) > 0

        return actions

    def compute_loss_cached(self, options, trajectory, joint_problem_name_list=None, problem_str=None, number_actions=3, cache_enabled=True):
        t = trajectory.get_trajectory()
        M = np.arange(len(t) + 1)
        trace = [(i-1, None) for i in range(len(t) + 1)]
        trace[0] = (0, None)
        used_sequences = set()

        for j in range(len(t) + 1):
            if j > 0:
                if M[j - 1] + 1 < M[j]:
                    trace[j] = (j - 1, None)
                    M[j] = M[j - 1] + 1
                # M[j] = min(M[j - 1] + 1, M[j])
            if j < len(t):
                # the mask being considered for selection cannot be evaluated on the trajectory
                # generated by the MLP trained to solve the problem.
                
                for i in range(len(options)):
                    option = options[i]
                    if option.problem_id == problem_str:
                        continue
                    # if j + number_steps[i] >= len(t):
                    #     continue
                    # if any([joint_problem_name_list[min(j+k, len(t) - 1)] == problem_str for k in range(1, option.option_size)]):
                    #     continue
                    option_id = option.get_option_id()
                    if option_id in self.cache and problem_str in self.cache[option_id] and j in self.cache[option_id][problem_str]:
                        if self.cache[option_id][problem_str][j][0] == True:
                            actions = self.cache[option_id][problem_str][j][1]
                            if M[j + len(actions)] > M[j] + 1:
                                trace[j + len(actions)] = (j, i)
                                M[j + len(actions)] = M[j] + 1
                            used_sequences.add((j, j+len(actions)))
                            # M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)
                    else:
                        if cache_enabled:
                            assert option_id in self.cache , f"{option_id} not found in cache"
                            assert problem_str in self.cache[option_id], f"{problem_str} not found in cache of {option_id}"
                            assert j in self.cache[option_id][problem_str], f"{j} not found in cache of {option_id} and {problem_str}"
                            raise Exception(f"The cache is supposed to be precomputed, combination wasn't found: \n {(option_id, problem_str, j)}")
                        else:
                            actions = self._run(copy.deepcopy(t[j][0]), option, option.option_size)
                            is_applicable = (len(actions) == option.option_size) and self.is_applicable(t, actions, j)
                            if is_applicable:
                                if M[j + len(actions)] > M[j] + 1:
                                    trace[j + len(actions)] = (j, i)
                                    M[j + len(actions)] = M[j] + 1
        uniform_probability = (1/(len(options) + number_actions)) 
        depth = len(t) + 1
        number_decisions = M[len(t)]

        option_usage = np.zeros(len(options))
        j = len(t)
        while j != 0:
            option_usage[trace[j][1]] += 1
            j = trace[j][0]
        reg = self.alpha * np.sum(1/option_usage[option_usage > 0])

        # use the Levin loss in log space to avoid numerical issues
        log_depth = math.log(depth)
        log_uniform_probability = math.log(uniform_probability)

        # self.logger.info(f"alpha potentially: {math.log(len(t)) - len(t) * math.log(1/number_actions) }")
        # self.logger.info(f"levin loss: {log_depth - number_decisions * log_uniform_probability}, reg: alpha* {np.sum(1/option_usage)}")

        return log_depth - number_decisions * log_uniform_probability + reg, used_sequences

    def print_output_subpolicy_trajectory(self, options: List[PPOAgent], trajectories, logger):
        """
        This function prints the "behavior" of the options encoded. It will show
        when each option is applicable in different states of the different trajectories. Here is 
        a typical output of this function.

        BL-TR
        Mask:  o0
        001001102102001102001102
        -----000----------------
        --------------000-------
        --------------------000-

        Mask:  o3
        001001102102001102001102
        ------333---------------
        ---------------333------
        ----------------333-----
        ---------------------333
        ----------------------33

        Number of Decisions:  18

        It shows how different options are used in a given sequence. In the example above, option o0
        is used in the sequence 110, while option o3 is used in some of the occurrences of 102. 
        """
        for idx, agent in enumerate(options):
            # Evaluating the performance of options
            logger.info(f"\n {idx} Option: {agent.problem_id}, {agent.extra_info}")

        for problem, trajectory in trajectories.items():  
            logger.info(f"Option Occurrences in {problem}")

            option_usage = {str(i): [] for i in range(len(options))}
            t = trajectory.get_trajectory()
            M = np.arange(len(t) + 1)

            for j in range(len(t) + 1):
                if j > 0:
                    if M[j - 1] + 1 < M[j]:
                        M[j] = M[j - 1] + 1

                if j < len(t):
                    for i in range(len(options)):

                        if options[i].problem_id == problem:
                            continue

                        actions = self._run(copy.deepcopy(t[j][0]), options[i], options[i].option_size)

                        if len(actions) == options[i].option_size and self.is_applicable(t, actions, j):
                            M[j + len(actions)] = min(M[j + len(actions)], M[j] + 1)

                            usage = ['-' for _ in range(len(t))]
                            for k in range(j, j+len(actions)):
                                usage[k] = str(i)
                            option_usage[str(i)].append(usage)
            
            for idx, matrix in option_usage.items():
                print(f"Option {idx} Occurrences:")
                buffer = "\n"
                for _, action in t:
                    buffer += str(action)
                buffer += "\n"
                for use in matrix:
                    for v in use:
                        buffer += str(v)
                    buffer += "\n"
                logger.info(buffer)
            logger.info(f'Number of Decisions:  {M[len(t)]}')

    def evaluate_on_each_cell(self, options: List[PPOAgent], trajectories: dict, problem_test, args, seed: int, logger=None):
        """
        This test is to see for each cell, options will give which sequence of actions
        """
        def _display_options(action_seq, game_width, indent=24):
            buffer = "\n"
            for i in range(game_width):
                for j in range(game_width):
                    if env.is_over(loc=(i,j)):
                        buffer += ("Goal" + " " * 6)
                        continue
                    buffer += (",".join(map(str, action_seq[(i,j)])) + " " * indent)[:indent]
                buffer += "\n"
            logger.info(buffer)

        env = get_single_environment(args, seed, problem=problem_test)
        game_width = args.game_width

        if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
            directions = ["R", "D", "L", "U"]
        elif args.env_id == "ComboGrid":
            directions = ["NA"]
        else:
            raise NotImplementedError
        
        for problem, t in trajectories.items():
            actions = t.get_action_sequence()
            t_str = [DIRECTIONS[tuple(actions[i:i+3])] for i in range(0, t.get_length(), 3)]
            print(f"Problem {problem}, model trajectory: {t_str}, {t.get_logits_sequence()}")
        
        for idx, agent in enumerate(options):
            # Evaluating the performance of options
            logger.info(f"\n {idx} Option: {agent.problem_id}, {agent.extra_info}")
            for direction in directions:
                logger.info(f"Direction: {direction}")
                action_seq = {}
                for i in range(game_width):
                    for j in range(game_width):    
                        env.reset(init_loc=(i,j), init_dir=direction)  
                        if env.is_over(loc=(i,j)):
                            continue
                                              
                        actions = self._run(env, agent, agent.option_size)
                        action_seq[(i,j)] = actions

                logger.info("Option Outputs:")
                _display_options(action_seq, game_width)

                # Evaluating the performance of original agents
                action_seq = {}
                for i in range(game_width):
                    for j in range(game_width):    
                        if env.is_over(loc=(i,j)):
                            continue
                        env.reset(init_loc=(i,j), init_dir=direction)
                        trajectory, _ = agent.run(env, length_cap=agent.option_size - 1)
                        actions = trajectory.get_action_sequence()
                        action_seq[(i,j)] = actions

                logger.info("Original Agent's Outputs:")
                _display_options(action_seq, game_width)
        logger.info("#### ### ###\n")
    

def main():
    args = process_args()

    # Logger configurations
    logger, args.log_path = utils.get_logger(args.exp_name, args.log_level, args.log_path)

    run_name = f'{args.exp_id}_sd{args.seed}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_id,
            job_type="eval",
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    buffer = "Parameters:\n"
    for key, value in vars(args).items():
        buffer += (f"{key}: {value}\n")
    logger.info(buffer)
    utils.logger_flush(logger)

    module_extractor = FineTuning(args, logger)
    module_extractor.discover()

    logger.info(f"Run id: {run_name}")
    logger.info(f"logs saved at {args.log_path}")


if __name__ == "__main__":
    main()