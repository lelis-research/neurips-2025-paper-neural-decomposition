import gymnasium as gym

from .Wrappers import WRAPPING_TO_WRAPPER

MINIGRID_ENV_LST = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0"
]

def get_single_env(env_name, max_steps=None, render_mode=None, env_params={}, wrapping_lst=None, wrapping_params=[]):
    """
    Create a single MiniGrid environment.
    
    Args:
        env_name (str): Name of the MiniGrid environment. Must be in MINIGRID_ENV_LST.
        max_steps (int): Maximum steps per episode.
        render_mode ("rgb_array" or "human" or None): Rendering mode for the environment.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper.
    
    Returns:
        gym.Env: A wrapped Gymnasium environment.
    """
    assert env_name in MINIGRID_ENV_LST, f"Environment {env_name} not supported."
    env = gym.make(env_name, max_steps=max_steps, render_mode=render_mode, **env_params)
    
    # Apply each wrapper in the provided list with corresponding parameters in the same order.
    for i, wrapper_name in enumerate(wrapping_lst):
        env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    
    return env


if __name__ == "__main__":
    env_wrapping= ["ViewSize", "FlattenOnehotObj", "StepReward"]
    wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}]
    env_params = {}
    env_name = "MiniGrid-Empty-5x5-v0"

    env = get_single_env(env_name, max_steps=500, render_mode="human", env_params=env_params, wrapping_lst=env_wrapping, wrapping_params=wrapping_params)
    
    obs, info = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()