import numpy as np
import gymnasium as gym

class RandomAgent:
    def __init__(self, observation_space, action_space):
       
        self.observation_space = observation_space
        self.action_space = action_space

              
    def act(self, observation, greedy=False):
        action = self.action_space.sample()
        return action

    def update(self, next_observation, reward, terminated, truncated):
        pass
    
    