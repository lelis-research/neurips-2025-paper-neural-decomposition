import gymnasium as gym
from gymnasium.envs.registration import register
from .Base import car_gym
from .Wrappers import WRAPPING_TO_WRAPPER

CAR_ENV_LST = [
    "car-train",
    "car-test"
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
    assert env_name in CAR_ENV_LST, f"Environment {env_name} not supported."
    env = gym.make(env_name, render_mode=render_mode, **env_params)
    
    # Apply each wrapper in the provided list with corresponding parameters in the same order.
    for i, wrapper_name in enumerate(wrapping_lst):
        env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    
    return env

register(
    id="car-train",
    entry_point="Environments.Car.Base.car_gym:CarEnv",
    kwargs={
        "test_mode": False
    },
)
register(
    id="car-test",
    entry_point="Environments.Car.Base.car_gym:CarEnv",
    kwargs={
        "test_mode": True
    },
)

if __name__ == "__main__":
    env_wrapping= []
    wrapping_params = []
    env_params = {}
    env_name = "car-train"

    env = get_single_env(env_name, render_mode="human", env_params=env_params, wrapping_lst=env_wrapping, wrapping_params=wrapping_params)
    
    obs, info = env.reset()
    for i in range(11000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()