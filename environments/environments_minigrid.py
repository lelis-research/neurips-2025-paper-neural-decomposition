import gymnasium as gym
import gymnasium
import copy
import torch
import numpy as np
from minigrid.wrappers import ViewSizeWrapper
from minigrid.core.world_object import Wall, Goal
from gymnasium.core import Wrapper
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.fourrooms import FourRoomsEnv
from minigrid.envs.unlock import UnlockEnv
from minigrid.wrappers import PositionBonus
from minigrid.core.roomgrid import RoomGrid
from environments.minigrid_multiroomunlock import MultiRoomUnlockEnv
from environments.environments_combogrid_gym import ComboGym
import copy
import os


class MiniGridWrap(gym.Env):
    def __init__(
        self,
        env,
        seed=None,
        n_discrete_actions=3,
        view_size=5,
        show_direction=False,
        options=None,
        step_reward=-1,
        goal_reward=1
    ):
        super(MiniGridWrap, self).__init__()
        # Define action and observation space
        self.seed_ = seed
        self.show_direction = show_direction
        self.env = env
        self.env = ViewSizeWrapper(env, agent_view_size=view_size)
        self.n_steps = 0
        # self.env.max_steps = max_episode_steps
        self.n_discrete_actions = n_discrete_actions
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.reset()
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        if options:
            self.setup_options(options)
        else:
            self.options = None

        shape = (len(self.get_observation()),)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float64
        )

        self.spec = self.env.spec
        self.agent_pos = self.env.unwrapped.agent_pos

    def setup_options(self, options):
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(options))
        self.options = copy.deepcopy(options)
        self.goal_reward = 10
        self.step_reward = 0

    def custom_reward(self, terminated):
        if terminated:
            return self.goal_reward
        else:
            return self.step_reward

    def one_hot_encode(self, observation):
        OBJECT_TO_ONEHOT = {
            0:  [0,0,0,0,0,0,0,0,0,0],  # unseen
            1:  [1,0,0,0,0,0,0,0,0,0],  # empty space
            2:  [0,1,0,0,0,0,0,0,0,0],  # wall
            3:  [0,0,1,0,0,0,0,0,0,0],  # floor
            4:  [0,0,0,1,0,0,0,0,0,0],  # door
            5:  [0,0,0,0,1,0,0,0,0,0],  # key
            6:  [0,0,0,0,0,1,0,0,0,0],  # ball
            7:  [0,0,0,0,0,0,1,0,0,0],  # box
            8:  [0,0,0,0,0,0,0,1,0,0],  # Goal
            9:  [0,0,0,0,0,0,0,0,1,0],  # lava
            10: [0,0,0,0,0,0,0,0,0,1],  # agent
        }
        one_hot = [OBJECT_TO_ONEHOT[int(x)] for x in observation]
        return np.array(one_hot).flatten()

    def get_observation(self):
        obs = self.env.unwrapped.gen_obs()
        self.agent_pos = self.env.unwrapped.agent_pos
        image = self.one_hot_encode(self.env.observation(obs)['image'][:,:,0].flatten())
        if isinstance(self.env.unwrapped, RoomGrid) or isinstance(self.env.unwrapped, MultiRoomUnlockEnv):
            image = np.concatenate((image, [1 if self.env.unwrapped.carrying is not None else 0]))
        if self.show_direction:
            return np.concatenate((
                image,
                self._dir_to_numeric(self.env.observation(obs)['direction'])
                # [self.agent_pos[0] - self.goal_position[0], self.agent_pos[1] - self.goal_position[1]]
            ))
        # return np.concatenate((image, [self.agent_pos[0] - self.goal_position[0], self.agent_pos[1] - self.goal_position[1]]))
        return image


    def _dir_to_numeric(self, direction: str):
        return {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}[direction]

    def step(self, action: int): 
        reward_sum = 0
        if self.options and action >= self.n_discrete_actions:
            option = self.options[action - self.n_discrete_actions]

            for idx in range(option.option_size):
                option_action, _ = option.get_action_with_mask(torch.tensor(self.get_observation(), dtype=torch.float32).view(1, -1))
                self.steps += 1
                _, temp_reward, terminated, truncated, _ = self.env.step(option_action)
                reward_sum += self.custom_reward(terminated)
                if terminated or truncated:
                    break
        else:
            idx = 0
            self.steps += 1
            _, temp_reward, terminated, truncated, _ = self.env.step(action)
            reward_sum = self.custom_reward(terminated)

        return (self.get_observation(), reward_sum, bool(terminated), bool(truncated), {"steps": self.steps, 'action_size': idx + 1})

    def reset(self, init_loc=None, init_dir:str=None, seed=None, options=None):
        self.steps = 0
        if seed is not None:
            self.seed_ = seed
        self.env.reset(seed=self.seed_)
        self.goal_position = [
            x for x, y in enumerate(self.env.unwrapped.grid.grid) if isinstance(y, Goal)
        ]
        if len(self.goal_position) > 0:
            self.goal_position = (
                int(self.goal_position[0] / self.env.unwrapped.height),
                self.goal_position[0] % self.env.unwrapped.height,
            )
        if init_loc and init_dir:
            self.env.unwrapped.agent_pos = np.array(init_loc)
            self.env.unwrapped.dir_init = self._dir_to_numeric(init_dir)
        self.agent_pos = self.env.unwrapped.agent_pos
        return self.get_observation(), {}

    def is_over(self, loc: tuple=None):
        if loc is None:
            loc = tuple(self.env.unwrapped.agent_pos)
        return loc == self.goal_position

    def render(self):
        return self.env.render()

    def seed(self, seed):
        self.seed_ = seed
        self.env.reset(seed=seed)

    def get_observation_space(self):
        return self.get_observation().size
    
    def get_action_space(self):
        return self.action_space.n

    def represent_options(self, options):
        str_map = ""
        for i in range(self.env.agent_view_size):
            for j in range(self.env.agent_view_size):
                if (i,j) in options:
                    str_map += ",".join([str(action) for action in options[(i,j)]]) + " "
                else:
                    str_map += "-,-,-"
            str_map += "\n"
        return str_map


def get_training_tasks_simplecross(view_size=7, seed=0, options=None):
    return MiniGridWrap(
                gymnasium.make("MiniGrid-SimpleCrossingS9N1-v0"),
                seed=seed, max_episode_steps=1000, n_discrete_actions=3,
                view_size=view_size, step_reward=-1, options=options)


def get_test_tasks_fourrooms(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=seed,
    )


def get_test_tasks_fourrooms2(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=51,
    )


def get_test_tasks_fourrooms3(view_size=7, seed=0):
    return MiniGridWrap(
            gymnasium.make("MiniGrid-FourRooms-v0"),
            max_episode_steps=19*19, n_discrete_actions=3, view_size=view_size, seed=41,
    )


def get_simplecross_env(*args, **kwargs):
    env = MiniGridWrap(
                CrossingEnv(obstacle_type=Wall, max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], seed=kwargs['seed']),
                seed=kwargs['seed'],
                n_discrete_actions=3,
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'])
    env.reset(seed=kwargs['seed'])
    if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
        env = PositionBonus(env, scale=0.001)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def get_fourrooms_env(*args, **kwargs):
    env = MiniGridWrap(
                env = FourRoomsEnv(max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], render_mode="rgb_array", see_through_walls=True),
                seed=kwargs['seed'],
                n_discrete_actions=3,
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'])
    env.reset(seed=kwargs['seed'])
    if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
        env = PositionBonus(env, scale=0.001)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def get_unlock_env(*args, **kwargs):
    env = MiniGridWrap(
                env=UnlockEnv(max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], render_mode="rgb_array"),
                seed=kwargs['seed'],
                n_discrete_actions=6 if 'n_discrete_actions' not in kwargs else kwargs['n_discrete_actions'],
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'])
    env.reset(seed=kwargs['seed'])
    if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
        env = PositionBonus(env, scale=0.001)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_env_simple_crossing(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                CrossingEnv(obstacle_type=Wall, max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], seed=kwargs['seed'], render_mode="rgb_array"),
                seed=kwargs['seed'],
                n_discrete_actions=3,
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'])
        env.reset(seed=kwargs['seed'])
        if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
            env = PositionBonus(env, scale=0.001)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_env_four_rooms(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                env = FourRoomsEnv(max_steps=361 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], render_mode="rgb_array", see_through_walls=True),
                seed=kwargs['seed'],
                n_discrete_actions=3,
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'],
                goal_reward=10,
                step_reward=0)
        env.reset(seed=kwargs['seed'])
        if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
            env = PositionBonus(env, scale=0.001)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_env_unlock(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                env = UnlockEnv(max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], render_mode="rgb_array"),
                seed=kwargs['seed'],
                n_discrete_actions=6 if 'n_discrete_actions' not in kwargs else kwargs['n_discrete_actions'],
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'])
        env.reset(seed=kwargs['seed'])
        if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
            env = PositionBonus(env, scale=0.001)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_env_multiroom(*args, **kwargs):
    def thunk():
        env = MiniGridWrap(
                env = MultiRoomUnlockEnv(max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], maxNumRooms=3, minNumRooms=3, render_mode="rgb_array", see_through_walls=True),
                seed=kwargs['seed'],
                n_discrete_actions=6 if 'n_discrete_actions' not in kwargs else kwargs['n_discrete_actions'],
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'],
                goal_reward=10,
                step_reward=0
                )
        env.reset(seed=kwargs['seed'])
        if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
            env = PositionBonus(env, scale=0.001)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def get_multiroom_env(*args, **kwargs):
    env = MiniGridWrap(
                env = MultiRoomUnlockEnv(max_steps=1000 if 'max_episode_steps' not in kwargs else kwargs['max_episode_steps'], maxNumRooms=3, minNumRooms=3, render_mode="rgb_array", see_through_walls=True),
                seed=kwargs['seed'],
                n_discrete_actions=6 if 'n_discrete_actions' not in kwargs else kwargs['n_discrete_actions'],
                view_size=kwargs['view_size'] if 'view_size' in kwargs else 9,
                show_direction=True if 'show_direction' not in kwargs else kwargs['show_direction'],
                options=None if 'options' not in kwargs else kwargs['options'],
                goal_reward=10,
                step_reward=0)
    env.reset(seed=kwargs['seed'])
    if 'visitation_bonus' in kwargs and kwargs['visitation_bonus'] == 1:
        env = PositionBonus(env, scale=0.001)
    return env