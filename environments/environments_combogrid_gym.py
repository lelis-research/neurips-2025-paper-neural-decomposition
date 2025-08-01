import copy
import gymnasium as gym
import numpy as np
import torch
from environments.environments_combogrid import Game, basic_actions
from typing import List, Any
from gymnasium.envs.registration import register

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=None, reward_per_step=-1, reward_goal=1, max_steps=500):
        self._game = Game(rows, columns, problem)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.n_discrete_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_discrete_actions)
        self.n_steps = 0
        self.reward_per_step = reward_per_step
        self.reward_goal = reward_goal
        
        if options is not None:
            self.setup_options(options)
        else:
            self.options = None

    def get_observation(self):
        return self._game.get_observation()
    
    def setup_options(self, options:List[Any]=None):
        """
        Enables the corresponding agents to choose from both actions and options
        """
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(options))
        self.options = copy.deepcopy(options)
    
    def reset(self, init_loc=None, init_dir=None, seed=0, options=None):
        self._game.reset(init_loc)
        self.n_steps = 0
        return self.get_observation(), {}
    
    def step(self, action:int):
        truncated = False
        def process_action(action: int):
            nonlocal truncated
            self._game.apply_action(action)
            self.n_steps += 1
            terminated, is_goal = self._game.is_over()
            reward = self.reward_goal if is_goal else self.reward_per_step 
            if self.n_steps == self.max_steps:
                truncated = True
            return self.get_observation(), reward, terminated, truncated, {"steps": self.n_steps, "action_size": 1}
    
        if self.options and action >= self.n_discrete_actions:
            reward_sum = 0
            option = self.options[action - self.n_discrete_actions]
            for idx in range(option.option_size):
                x_tensor = torch.tensor(self.get_observation(), dtype=torch.float32).view(1, -1)
                if option.mask is not None:
                    option_action, _ = option.get_action_with_mask(x_tensor)
                else:
                    option_action = option.get_action_and_value(x_tensor, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = process_action(option_action)
                reward_sum += reward
                if terminated or truncated:
                    return obs, reward_sum, terminated, truncated, {"steps": self.n_steps, "action_size": idx + 1}
            return obs, reward_sum, terminated, truncated, {"steps": self.n_steps, "action_size": idx + 1}
        else:
            return process_action(action)
    
    def is_over(self, loc=None):
        if loc:
            return any([loc == goal for goal in self._game.get_goals()])
        return self._game.is_over()[0]
    
    def get_observation_space(self):
        return self._rows * self._columns * 2 + 9
    
    def get_action_space(self):
        return self.action_space.n
    
    def represent_options(self, options):
        return self._game.represent_options(options)
    

def make_env(*args, **kwargs):
    def thunk():
        env = ComboGym(*args, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


register(
     id="ComboGridWorld-v0",
     entry_point=ComboGym
)