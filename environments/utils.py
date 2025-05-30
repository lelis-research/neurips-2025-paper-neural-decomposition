import gymnasium as gym
from environments.environments_combogrid_gym import ComboGym, make_env as make_env_combogrid
from environments.environments_minigrid import get_simplecross_env, make_env_four_rooms, make_env_simple_crossing, make_env_unlock
from environments.environments_combogrid import PROBLEM_NAMES as COMBO_PROBLEM_NAMES

def get_single_environment(args, seed, problem=None, is_test=False, options=None):
    """
    Makes a single environment based on `args.env_id`

    Parameters:
        args (Args): Requires to include `args.env_id`.
        seed (int).
        problem (str): overrides seed for `ComboGrid`.
        is_test (bool): determines if the environment is set to test or not, changing configurations e.g. rewards.
    """
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        # requires `args` to include `args.game_width`
        env = get_simplecross_env(view_size=args.game_width, seed=seed, options=options)
    elif args.env_id == "ComboGrid":
        # requires `args` to include `args.game_width`
        if not problem:
            problem = COMBO_PROBLEM_NAMES[seed]
        reward_per_step = 0
        reward_goal = 10 if is_test else 1
        env = ComboGym(rows=args.game_width, 
                       columns=args.game_width, 
                       problem=problem, 
                       options=options,
                       reward_per_step=reward_per_step, 
                       reward_goal=reward_goal)
    else:
        raise NotImplementedError
    return env


def get_single_environment_builder(args, seed, problem=None, options=None, is_test=False):
    """
    Makes a single environment based on `args.env_id`

    Parameters:
        args (Args): Requires to include `args.env_id`.
        seed (int).
        problem (str): overrides seed for `ComboGrid`.
        options (list).
        is_test (bool): determines if the environment is set to test or not, changing configurations e.g. rewards.
    """
    if args.env_id == "MiniGrid-SimpleCrossingS9N1-v0":
        # requires `args` to include `args.game_width`
        env_fn = make_env_simple_crossing(view_size=args.game_width, seed=seed, options=options)
    elif "ComboGrid" in args.env_id:
        if not problem:
            problem = COMBO_PROBLEM_NAMES[seed]
        reward_per_step = 0
        reward_goal = 10 if is_test else 1
        env_fn = make_env_combogrid(rows=args.game_width, 
                                    columns=args.game_width, 
                                    problem=problem, 
                                    reward_per_step=reward_per_step, 
                                    reward_goal=reward_goal,
                                    options=options)
    elif args.env_id == "MiniGrid-FourRooms-v0":
        env_fn = make_env_four_rooms(view_size=args.game_width, seed=seed, options=options)
    elif args.env_id == "MiniGrid-Unlock-v0":
        env_fn = make_env_unlock(view_size=args.game_width, seed=seed, options=options)
    else:
        raise NotImplementedError
    return env_fn