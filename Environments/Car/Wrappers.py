import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from gymnasium.wrappers import NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, ClipAction
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper

# RewardWrapper that adds a constant step reward to the environment's reward.
class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=-0.01):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward

class ClipReward(RewardWrapper):
    def __init__(self, env, min_reward=-100, max_reward=100):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
    
    
class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info
    

class WarmupRewardNormalizer(gym.Wrapper):
    """
    Like NormalizeReward but with a warm-up:
      - First `warmup_steps` → return raw rewards
      - Thereafter      → return (r - μ)/(√σ² + ε)
    """
    def __init__(self, env, gamma=0.99, epsilon=1e-8, warmup_steps=10000):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.warmup = warmup_steps
        self.step_count = 0

        # EMA stats on *raw* rewards
        self.mean = 0.0
        self.var  = 1.0  # start with var=1 to avoid div-by-zero

        # for computing discounted‐return estimate exactly like NormalizeReward
        self.discounted_reward = 0.0

    def step(self, action):
        # 1) step raw env
        obs, raw_r, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # 2) update discounted‐return and EMA stats
        #    same as gym's NormalizeReward
        self.discounted_reward = (
            self.discounted_reward * self.gamma * (1 - float(terminated))
            + float(raw_r)
        )
        # update mean/var on discounted_reward
        delta = self.discounted_reward - self.mean
        self.mean += (1 - self.gamma) * delta
        self.var  = self.gamma * self.var + (1 - self.gamma) * (delta * delta)

        # 3) decide raw vs normalized
        if self.step_count <= self.warmup:
            return obs, raw_r, terminated, truncated, info
        else:
            norm_r = raw_r / (np.sqrt(self.var) + self.epsilon)
            return obs, norm_r, terminated, truncated, info
    
# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "NormalizeObs": NormalizeObservation,
    "ClipObs": TransformObservation,
    "NormalizeReward": NormalizeReward,
    "ClipReward": ClipReward,
    "RecordReward": RecordRewardWrapper,
    "ClipAction": ClipAction,
    "StepReward":StepRewardWrapper,
    "WarmupRewardNormalizer": WarmupRewardNormalizer,
}