import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from gymnasium.wrappers import NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, ClipAction
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper

# RewardWrapper that adds a constant step reward to the environment's reward.
class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=0):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward
    
class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info
    
# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "NormalizeObs": NormalizeObservation,
    "ClipObs": TransformObservation,
    "NormalizeReward": NormalizeReward,
    "ClipReward": TransformReward,
    "RecordReward": RecordRewardWrapper,
    "ClipAction": ClipAction,
    "StepReward":StepRewardWrapper,
}