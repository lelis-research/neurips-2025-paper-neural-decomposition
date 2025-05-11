import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.envs.registration import register

from .Wrappers import WRAPPING_TO_WRAPPER

MUJOCO_ENV_LST = [
    "Maze_1m",
    "Maze_2m",
    "Maze_3m",
    "Maze_4m",
    
    "Maze_1e",
    "Maze_2e",
    "Maze_3e",
    "Maze_4e",
    
    "Maze_R",
    "Maze_L",
    "Maze_U",
    "Maze_D",
    
    "Medium_Maze",
    "Large_Maze",
    "Hard_Maze",
    "Tunnel_Maze",

    "AntMaze_R",
    "AntMaze_L",
    "AntMaze_U",
    "AntMaze_D",

    "AntMaze_UMaze-v5",
    "AntMaze_UMazeDense-v5",
    
    "Ant-v5",
]


def get_single_env(env_name, max_steps=None, render_mode=None, env_params={}, wrapping_lst=None, wrapping_params=[]):
    """
    Create a single MiniGrid environment.
    
    Args:
        env_name (str): Name of the MiniGrid environment. Must be in MINIGRID_ENV_LST.
        render_mode ("rgb_array" or "human" or None): Rendering mode for the environment.
        wrapping_lst (list or None): List of wrapper names to apply.
        wrapping_params (list): List of parameter dictionaries for each wrapper.
    
    Returns:
        gym.Env: A wrapped Gymnasium environment.
    """
    assert env_name in MUJOCO_ENV_LST, f"Environment {env_name} not supported."
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_name, max_episode_steps=max_steps, render_mode=render_mode, **env_params)
    
    # Apply each wrapper in the provided list with corresponding parameters in the same order.
    for i, wrapper_name in enumerate(wrapping_lst):
        if wrapper_name == "ClipObs":
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, observation_space=env.observation_space, **wrapping_params[i])
        else:
            env = WRAPPING_TO_WRAPPER[wrapper_name](env, **wrapping_params[i])
    
    return env

R, G = 'r', 'g'
maze_r =[
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1, 'r', 0,'g',1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
]
maze_l =[
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1, 'g', 0,'r',1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
]
maze_u =[
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1, 'g', 1, 1,  1],
    [ 1, 1,  1,  0, 1, 1,  1],
    [ 1, 1,  1, 'r', 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
]
maze_d =[
    [ 1, 1,  1,  1, 1, 1,  1],
    [ 1, 1,  1, 'r', 1, 1,  1],
    [ 1, 1,  1,  0, 1, 1,  1],
    [ 1, 1,  1, 'g', 1, 1,  1],
    [ 1, 1,  1,  1, 1, 1,  1],
]


maze_1e = [
    [ 1, 1, 1, 1, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, R, 1, G, 1],
    [ 1, 1, 1, 1, 1],
]

maze_2e = [
    [ 1, 1, 1, 1, 1],
    [ 1, R, 1, G, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 1, 1, 1, 1],
]
maze_3e = [
    [ 1, 1, 1, 1, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, G, 1, R, 1],
    [ 1, 1, 1, 1, 1],
]
maze_4e = [
    [ 1, 1, 1, 1, 1],
    [ 1, G, 1, R, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 1, 1, 1, 1],
]

maze_1m = [
    [ 1, 1, 1, 1, 1],
    [ 1, R, 1, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 1, G, 1],
    [ 1, 1, 1, 1, 1],
]
maze_2m = [
    [ 1, 1, 1, 1, 1],
    [ 1, 0, 1, G, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, R, 1, 0, 1],
    [ 1, 1, 1, 1,1 ],
]
maze_3m = [
    [ 1, 1, 1, 1, 1],
    [ 1, G, 1, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 1, R, 1],
    [ 1, 1, 1, 1, 1],
]
maze_4m = [
    [ 1, 1, 1, 1, 1],
    [ 1, 0, 1, R, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, 0, 0, 0, 1],
    [ 1, 0, 1, 0, 1],
    [ 1, G, 1, 0, 1],
    [ 1, 1, 1, 1, 1],
]


medium_maze = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
large_maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]    
]
hard_maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, G, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

tunnel_maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, G, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]




register(
    id="Medium_Maze",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": medium_maze,
    },
)
register(
    id="Large_Maze",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": large_maze,
    },
)
register(
    id="Hard_Maze",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": hard_maze,
    },
)
register(
    id="Tunnel_Maze",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": tunnel_maze,
    },
)

register(
    id="Maze_R",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_r,
    },
)
register(
    id="Maze_L",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_l,
    },
)
register(
    id="Maze_U",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_u,
    },
)
register(
    id="Maze_D",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_d,
    },
)
#**
register(
    id="Maze_1e",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_1e,
    },
)
register(
    id="Maze_2e",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_2e,
    },
)
register(
    id="Maze_3e",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_3e,
    },
)
register(
    id="Maze_4e",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_4e,
    },
)
#**
register(
    id="Maze_1m",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_1m,
    },
)
register(
    id="Maze_2m",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_2m,
    },
)
register(
    id="Maze_3m",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_3m,
    },
)
register(
    id="Maze_4m",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "maze_map": maze_4m,
    },
)


#AntMaze Sparse
register(
    id="AntMaze_R",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    kwargs={
        "maze_map": maze_r,
    },
)
register(
    id="AntMaze_L",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    kwargs={
        "maze_map": maze_l,
    },
)
register(
    id="AntMaze_U",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    kwargs={
        "maze_map": maze_u,
    },
)
register(
    id="AntMaze_D",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    kwargs={
        "maze_map": maze_d,
    },
)



if __name__ == "__main__":
    env_wrapping= [
        "CombineGoals",
        "ClipAction",
        "NormalizeObs",
        "ClipObs",
        "RecordReward",
        "NormalizeReward",
        "ClipReward", 
        ]
    
    wrapping_params = [
        {},
        {}, {},
        {"func": lambda obs: np.clip(obs, -10, 10)}, 
        {}, {},
        {"func": lambda reward: np.clip(reward, -10, 10)},
        ]
    env_params = {}
    env_name = "AntMaze_UMaze-v5"

    env = get_single_env(env_name, render_mode="human", env_params=env_params, wrapping_lst=env_wrapping, wrapping_params=wrapping_params)
    
    obs, info = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Obs shape: {obs.shape}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()