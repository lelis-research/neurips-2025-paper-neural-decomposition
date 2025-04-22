import gymnasium as gym
import numpy as np
from gymnasium.wrappers import NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, ClipAction
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper

# Record the original reward in the info dict.
# Useful for when using the reward normalizer wrapper (use before normalizer).
class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info

class CombineGoalsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_space = self.env.observation_space
        assert isinstance(original_space, gym.spaces.Dict)

        obs_dim = original_space["observation"].shape[0]
        goal_dim = original_space["desired_goal"].shape[0]
        # achieved_dim = original_space["achieved_goal"].shape[0]

        total_dim = obs_dim + goal_dim #+ achieved_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )
    def observation(self, observation):
        return np.concatenate([
            observation["observation"],
            observation["desired_goal"],
            # observation["achieved_goal"],
        ], axis=0)

class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=-1.0):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward
    
# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "CombineGoals": CombineGoalsWrapper,
    "NormalizeObs": NormalizeObservation,
    "ClipObs": TransformObservation,
    "NormalizeReward": NormalizeReward,
    "ClipReward": TransformReward,
    "RecordReward": RecordRewardWrapper,
    "ClipAction": ClipAction,
    "StepReward":StepRewardWrapper,
}

