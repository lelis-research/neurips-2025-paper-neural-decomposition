import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium.envs.registration import register

from .Wrappers import WRAPPING_TO_WRAPPER

MUJOCO_ENV_LST = [
    "Maze_1_Sparse",
    "Maze_2_Sparse",
    "Maze_3_Sparse",
    "Maze_4_Sparse",
    "Maze_Test_Sparse",

    "Maze_1_Dense",
    "Maze_2_Dense",
    "Maze_3_Dense",
    "Maze_4_Dense",

    "Maze_R",
    "Maze_L",
    "Maze_U",
    "Maze_D",
    "Medium_Maze_Sparse",

    "AntMaze_1_Sparse",
    "AntMaze_2_Sparse",
    "AntMaze_3_Sparse",
    "AntMaze_4_Sparse",

    "AntMaze_1_Dense",
    "AntMaze_2_Dense",
    "AntMaze_3_Dense",
    "AntMaze_4_Dense",

    "AntMaze_R",
    "AntMaze_L",
    "AntMaze_U",
    "AntMaze_D",

    "AntMaze_UMaze-v5",
    "AntMaze_UMazeDense-v5",

    "PointMaze_UMaze-v3",
    "PointMaze_UMazeDense-v3",

    
    "Walker2d-v4", 
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


maze_1 = [
    [ 1,   1,   1,   1,  1],
    [ 1,   0,   0,  'g', 1],
    [ 1,   0,   0,   0,  1],
    [ 1,  'r',  0,   0,  1],
    [ 1,   1,   1,   1,  1],
]
maze_2 = [
    [ 1,   1,   1,   1,  1],
    [ 1,  'r',  0,   0,  1],
    [ 1,   0,   0,   0,  1],
    [ 1,   0,   0,  'g', 1],
    [ 1,   1,   1,   1,  1],
]
maze_3 = [
    [ 1,   1,   1,   1,  1],
    [ 1,   0,   0,  'r', 1],
    [ 1,   0,   0,   0,  1],
    [ 1,  'g',  0,   0,  1],
    [ 1,   1,   1,   1,  1],
]
maze_4 = [
    [ 1,   1,   1,   1,  1],
    [ 1,  'g',  0,   0,  1],
    [ 1,   0,   0,   0,  1],
    [ 1,   0,   0,  'r', 1],
    [ 1,   1,   1,   1,  1],
]

# maze_test = [
#     [ 1,   1,   1,   1,  1,  1],
#     [ 1,  'c',  0,   0, 'c', 1],
#     [ 1,   0,   0,   0,  0,  1],
#     [ 1,   0,   0,   0,  0,  1],
#     [ 1,  'c',  0,   0, 'c', 1],
#     [ 1,   1,   1,   1,  1,  1],
# ]
maze_test = [
    [1, 1,  1, 1, 1, 1, 1, 1],
    [1,'r', 0, 1, 1, 0, 0, 1],
    [1, 0,  0, 1, 0, 0, 0, 1],
    [1, 1,  0, 0, 0, 1, 1, 1],
    [1, 0,  0, 1, 0, 0, 0, 1],
    [1, 0,  1, 0, 0, 1, 0, 1],
    [1, 0,  0, 0, 1,'g',0, 1],
    [1, 1,  1, 1, 1, 1, 1, 1],
]
#Maze Sparse
register(
    id="Maze_1_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_1,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_2_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_2,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_3_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_3,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_4_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_4,
        "reward_type": "sparse",   # or "dense"
    },
)

#Maze Dense
register(
    id="Maze_1_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_1,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="Maze_2_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_2,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="Maze_3_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_3,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="Maze_4_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_4,
        "reward_type": "dense",   # or "dense"
    },
)

register(
    id="Medium_Maze_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_map": maze_test,
        "reward_type": "sparse",   # or "dense"
    },
)

register(
    id="Maze_R",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_r,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_L",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_l,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_U",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_u,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="Maze_D",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_d,
        "reward_type": "sparse",   # or "dense"
    },
)


#AntMaze Sparse
register(
    id="AntMaze_R",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_r,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_L",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_l,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_U",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_u,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_D",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_d,
        "reward_type": "sparse",   # or "dense"
    },
)


register(
    id="AntMaze_1_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_1,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_2_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_2,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_3_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_3,
        "reward_type": "sparse",   # or "dense"
    },
)
register(
    id="AntMaze_4_Sparse",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_4,
        "reward_type": "sparse",   # or "dense"
    },
)

register(
    id="AntMaze_1_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_1,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="AntMaze_2_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_2,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="AntMaze_3_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_3,
        "reward_type": "dense",   # or "dense"
    },
)
register(
    id="AntMaze_4_Dense",                                    # your custom name
    entry_point="gymnasium_robotics.envs.maze.ant_maze_v3:AntMazeEnv",
    max_episode_steps=200,
    kwargs={
        "maze_map": maze_4,
        "reward_type": "dense",   # or "dense"
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