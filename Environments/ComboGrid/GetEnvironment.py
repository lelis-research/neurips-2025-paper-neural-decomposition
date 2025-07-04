import gymnasium as gym
from .environments_combogrid import PROBLEM_NAMES
from .environments_combogrid_gym import ComboGym

from .Wrappers import WRAPPING_TO_WRAPPER

COMBOGRID_ENV_LST = [
    "ComboGrid",
    "ComboGrid4",
]

def get_single_env(env_name, max_steps=None, render_mode=None, env_params={}, wrapping_lst=None, wrapping_params=[]):
    """
    Create a single ComboGrid environment.
    
    Args:
        env_name (str): Name of the ComboGrid environment. Must be in ComboGrid.
        max_steps (int): Maximum steps per episode.
        render_mode ("rgb_array" or "human" or None): Rendering mode for the environment.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper.
    
    Returns:
        gym.Env: A wrapped Gymnasium environment.
    """
    assert env_name in COMBOGRID_ENV_LST, f"Environment {env_name} not supported."

    action_pattern_length = 3
    if env_name == "ComboGrid4":
        action_pattern_length = 4

    problem = PROBLEM_NAMES[env_params['env_seed']]
    reward_per_step = env_params['step_reward']
    reward_goal = env_params['goal_reward']
    game_width = env_params['game_width']
    env = ComboGym(rows=game_width, 
                    columns=game_width, 
                    problem=problem, 
                    reward_per_step=reward_per_step, 
                    reward_goal=reward_goal,
                    max_steps=max_steps,
                    action_pattern_length=action_pattern_length)
        
    # Apply each wrapper in the provided list with corresponding parameters in the same order.
    # for i, wrapper_name in enumerate(wrapping_lst):
    #     env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    
    return env


if __name__ == "__main__":
    env_wrapping= ["ViewSize", "FlattenOnehotObj", "StepReward"]
    wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}]
    env_params = {}
    env_name = "COMBOGRID-Empty-5x5-v0"

    env = get_single_env(env_name, max_steps=500, render_mode="human", env_params=env_params, wrapping_lst=env_wrapping, wrapping_params=wrapping_params)
    
    obs, info = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()