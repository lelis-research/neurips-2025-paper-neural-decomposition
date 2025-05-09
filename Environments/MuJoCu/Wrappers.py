import gymnasium as gym
import numpy as np
import random
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

class ExtractObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_space = self.env.observation_space
        assert isinstance(original_space, gym.spaces.Dict)

        obs_dim = original_space["observation"].shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
    def observation(self, observation):
        return np.concatenate([
            observation["observation"],
        ], axis=0)
    
class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=-1.0):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward
    
class SuccessBonus(gym.Wrapper):
    def __init__(self, env, bonus=+5.0, threshold=0.5):
        super().__init__(env)
        self.bonus = bonus
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # extract achieved and desired goal from the observation dict
        ag = obs.get("achieved_goal")
        dg = obs.get("desired_goal")

        if ag is not None and dg is not None:
            distance = np.linalg.norm(ag - dg)
            if distance < self.threshold:
                reward += self.bonus
                info["success"] = True
                reward += self.bonus
                print("success!!")
            else:
                info["success"] = False

        return obs, reward, terminated, truncated, info
    
class RewardShaping(gym.Wrapper):
    def step(self, action):
        obs, r, terminated, truncated, info = super().step(action)
        r += info['reward_forward'] + info['reward_ctrl'] + info['reward_ctrl'] 

        return obs, r, terminated, truncated, info

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, min_radius=1, max_radius=3, step=1):
        super().__init__(env)
        maze_map = self.env.unwrapped.maze._maze_map
        self.min_r, self.max_r, self.r_step = min_radius, max_radius, step
        self.curr_r = min_radius

        # build lists of cells by type
        self.reset_cells = []
        self.free_cells  = []
        self.goal_cells  = []
        M = np.array(maze_map)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i,j]
                if v == '0':        self.free_cells.append((i,j))
                if v in ('r','c'):   self.reset_cells.append((i,j))
                if v in ('g','c'):   self.goal_cells.append((i,j))

    def reset(self, **kwargs):
        # pick a random reset cell
        start = random.choice(self.reset_cells)

        # if still ramping, pick a goal at manhattan‐distance == curr_r
        if self.curr_r < self.max_r:
            # candidates at exact manhattan distance
            cands = [
                cell for cell in self.free_cells + self.goal_cells
                if abs(cell[0]-start[0]) + abs(cell[1]-start[1]) == self.curr_r
            ]
            # fallback to any free/goal cell if none exactly match
            goal = random.choice(cands) if cands else random.choice(self.free_cells + self.goal_cells)
            
        else:
            # final stage: always use *your* original 'g' cells
            goal = random.choice(self.goal_cells)

        kwargs.pop("options", None)
        obs, info = self.env.reset(options={
            "reset_cell": np.array(start, dtype=int),
            "goal_cell":  np.array(goal,  dtype=int),
        }, **kwargs)
        
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        # adjust difficulty only at episode end
        if done and info.get("success", False):
            self.curr_r = min(self.curr_r + self.r_step, self.max_r)
        elif done:
            self.curr_r = max(self.curr_r - self.r_step, self.min_r)

        return obs, reward, term, trunc, info
    
    
class AntV5RewardWrapper(gym.Wrapper):
    """
    Wrap any MuJoCo Ant env (e.g. AntMaze_UMaze-v5) and
    use Ant-v5’s native reward signal instead of the maze’s.
    """
    def __init__(self, env, ant_r_coef=0.1):
        super().__init__(env)
        # build a parallel Ant-v5 env for its reward
        self.ant = gym.make("Ant-v5")
        self.ant_r_coef = ant_r_coef

    def reset(self, **kwargs):
        # reset both environments
        obs, info = super().reset(**kwargs)
        self.ant.reset()
        return obs, info

    def step(self, action):
        # 1) step your wrapped env (we’ll keep its reward in info)
        obs, maze_r, terminated, truncated, info = super().step(action)
        info['maze_reward'] = maze_r

        # 2) step the Ant-v5 env for its native reward
        _, ant_r, term2, trunc2, ant_info = self.ant.step(action)

        # merge any Ant-info you want
        info.update(ant_info)
        info['ant_reward'] = ant_r
        r = self.ant_r_coef * ant_r + maze_r

        # 3) return the Ant-v5 reward (ignore maze_r in the returned reward)
        return obs, ant_r, terminated, truncated, info
    
    
# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "CombineGoals": CombineGoalsWrapper,
    "ExtractObs": ExtractObsWrapper,
    "NormalizeObs": NormalizeObservation,
    "ClipObs": TransformObservation,
    "NormalizeReward": NormalizeReward,
    "ClipReward": TransformReward,
    "RecordReward": RecordRewardWrapper,
    "ClipAction": ClipAction,
    "StepReward": StepRewardWrapper,
    "SuccessBonus": SuccessBonus,
    "CurriculumWrapper": CurriculumWrapper,
    "RewardShaping": RewardShaping,
    "AntReward": AntV5RewardWrapper,
}

