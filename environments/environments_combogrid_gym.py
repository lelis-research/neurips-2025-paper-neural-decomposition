import copy
import gymnasium as gym
import numpy as np
import torch
from environments.environments_combogrid import Game, basic_actions
from typing import List, Any
from gymnasium.envs.registration import register

class ComboGym(gym.Env):
    def __init__(self, rows=3, columns=3, problem="TL-BR", options=None, max_length=500, visitation_bonus=False):
        self._game = Game(rows, columns, problem, visitation_bonus=visitation_bonus)
        self._rows = rows
        self._columns = columns
        self._problem = problem
        self.render_mode = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self._game.get_observation()), ), dtype=np.float64)
        self.n_discrete_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_discrete_actions)
        self.n_steps = 0
        self.max_length = max_length
        
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
        self.info = {'n_steps': 0, "goals": 0}
        return self.get_observation(), {}
    
    def step(self, action:int):
        truncated = False
        def process_action(action: int):
            nonlocal truncated
            visitation_reward = self._game.apply_action(action)
            # info["actions"].append(action)
            # info["observations"].append(self.get_observation())
            self.n_steps += 1
            terminated, reached_goal = self._game.is_over()
            if reached_goal:
                self.info["goals"] += 1
            if self.options:
                reward = 10 if reached_goal else 0
            else:
                reward = 1 if reached_goal else -1
            reward += visitation_reward
            if self.n_steps == self.max_length:
                truncated = True
            return self.get_observation(), reward, terminated, truncated, {}
    
        if self.options and action >= self.n_discrete_actions:
            reward_sum = 0
            option = self.options[action - self.n_discrete_actions]
            gru_state = option.init_hidden().squeeze(0)
            for _ in range(option.option_size):
                option_action, _, gru_state = option._get_action_with_input_mask_softmax(x_tensor=torch.tensor(self.get_observation(), dtype=torch.float32).view(1, -1), gru_state=gru_state)
                obs, reward, terminated, truncated, _ = process_action(option_action)
                reward_sum += reward
                if terminated or truncated:
                    self.info['n_steps'] = self.n_steps
                    return obs, reward_sum, terminated, truncated, self.info
            self.info['n_steps'] = self.n_steps
            return obs, reward_sum, terminated, truncated, self.info
        else:
            obs, reward, terminated, truncated, _ = process_action(action)
            self.info['n_steps'] = self.n_steps
            return obs, reward, terminated, truncated, self.info
    
    def is_over(self, loc=None):
        if loc:
            return loc == self._game.problem.goal
        return self._game.is_over()
    
    def get_observation_space(self):
        return len(self.get_observation())
    
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