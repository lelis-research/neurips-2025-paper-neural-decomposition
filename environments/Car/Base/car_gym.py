import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pathlib
import argparse
import time
import matplotlib.pyplot as plt

from .car_simulation import CarReversePP


class CarEnv(gym.Env):
    """
    A Gymnasium environment wrapper for the CarReversePP simulation.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, n_steps=5000, render_mode=None, test_mode=False, last_state_in_obs=True):
        super().__init__()
        self.sim = CarReversePP(n_steps=n_steps)
        self.render_mode = render_mode
        self.test_mode = test_mode
        self.last_state_in_obs = last_state_in_obs

        self.n_steps = n_steps
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
        )
        if self.last_state_in_obs:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)

        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float64)

    def step(self, action):
        self.counter += 1

        prev_state = np.copy(self.state)
        self.state = self.sim.simulate(self.state, action, self.sim.dt)

        goal_err = np.sum(self.sim.check_goal(self.state))
        safe_err = self.sim.check_safe(self.state)

        reward = - goal_err #- 10 * (safe_err > 0)
        truncated = self.counter >= self.n_steps
        terminated = goal_err < 0.1
        if terminated:
            print("yay")
        
        # safe_error > 0.05 --- goal_error < 0.01
     
        if self.last_state_in_obs:
            observation = np.concatenate([self.sim.get_features(self.state), self.sim.get_features(prev_state)])
        else:
            observation = self.sim.get_features(self.state)
        
        if self.render_mode == 'human':
            self.render()
            
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset_render()
        self.counter = 0

        if self.test_mode:
            test_limit = (11, 12)
            self.sim.set_inp_limits(test_limit)
        else:
            train_limit = (12, 13.5)
            self.sim.set_inp_limits(train_limit)
        
        self.state = self.sim.sample_init_state()

        if self.last_state_in_obs:
            return np.concatenate([self.sim.get_features(self.state), self.sim.get_features(self.state)]), {}
        
        return self.state, {}

    def render(self):
        if self.render_mode in ('human', 'rgb_array'):
            return self.sim.render(self.state, self.render_mode)
        return None

    def close(self):
        self.sim.reset_render()


def make_car_env(max_episode_steps=10000):
    def thunk():
        env = CarEnv(n_steps=max_episode_steps, 
                     render_mode=None, 
                     test_mode=False,
                     last_state_in_obs=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run CarEnv in either simulation or test mode.")

    parser.add_argument('--mode', type=str, default='gym', choices=['gym', 'test'], help="Choose 'gym' for simulation or 'test' for trajectory plotting.")
    parser.add_argument("--video_prefix", type=str, default="eval", help="Prefix for the video file name")
    args = parser.parse_args()

    if args.mode == 'gym':

        video_dir = str(pathlib.Path(__file__).parent.resolve() / "videos")
        os.makedirs(video_dir, exist_ok=True)

        env = CarEnv(n_steps=200, render_mode='rgb_array', test_mode=False)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode: episode == 0,
            name_prefix="parking_" + args.video_prefix,
        )
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        state_action_list = []
        collision_states = []
        while not done:

            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs[:4]

            # Store collision for plotting
            collision = env.sim.check_safe(state)
            if collision > 0:
                collision_states.append(state)

            # Store states for plotting
            state_action_list.append((state, action))

            total_reward += reward
            done = terminated or truncated
            
            time.sleep(env.sim.dt)
            
            if terminated or truncated:
                break

        print(f"Total Reward: {total_reward}")
        env.close()

        ###### Plotting the trajectory ######
        sim_plot = CarReversePP()
        plt.figure(figsize=(4, 8))

        start_state = [state_action_list[0][0]]       
        goal_state  = [state_action_list[-1][0]]       
        sim_plot.plot_init_paper(start_state, goal_state)
        sim_plot.plot_states(state_action_list, line=True)
        sim_plot.plot_collision_states(collision_states)

        plt.title("Sample Trajectory")
        plt.legend()
        plt.show()


    # Test in the original code
    elif args.mode == 'test':

        from math import pi
        def simulate_bicycle(state, action, dt):
            ns = np.copy(state)
            v, w = action 
            w = w / 10.0
            x, y, ang = ns  
            beta = np.arctan(0.5 * np.tan(w))
            dx = v * np.cos(ang + beta) * dt 
            dy = v * np.sin(ang + beta) * dt 
            da = v / 2.5 * np.sin(beta) * dt 
            ns[0] += dx 
            ns[1] += dy 
            ns[2] += da 
            return ns 

        def get_all_vertices(x, y, ang, w, h):
            res = []
            db = w / 2.0
            da = h / 2.0
            coa = np.cos(ang)
            sia = np.sin(ang)
            res.append((x + da * coa + db * sia, y + da * sia - db * coa))
            res.append((x + da * coa - db * sia, y + da * sia + db * coa))
            res.append((x - da * coa - db * sia, y - da * sia + db * coa))
            res.append((x - da * coa + db * sia, y - da * sia - db * coa))
            return res 

        def get_traj(s, a, T, w=1.8, h=5.0):
            X = []
            Y = []
            X1 = []
            Y1 = []  
            
            vertices = get_all_vertices(s[0], s[1], s[2], w, h)
            X.append(s[0])
            Y.append(s[1])
            # Using the average of the bottom vertices for one set of points
            X1.append((vertices[2][0] + vertices[3][0]) / 2.0)
            Y1.append((vertices[2][1] + vertices[3][1]) / 2.0)

            for i in range(T):
                s = simulate_bicycle(s, a, 0.01)
                vertices = get_all_vertices(s[0], s[1], s[2], w, h)
                X.append(s[0])
                Y.append(s[1])
                X1.append((vertices[2][0] + vertices[3][0]) / 2.0)
                Y1.append((vertices[2][1] + vertices[3][1]) / 2.0)
            
            color = 'g' if a[0] > 0 else 'r'
            plt.plot(X, Y, color, label="Car center" if i == 0 else "")
            plt.plot(X1, Y1, color+"--", label="Rear mid-point" if i == 0 else "")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Test Trajectory Plot")
            plt.legend()
            return s 

        # Set initial state: [x, y, angle]
        s = np.array([0.0, 0.0, pi/2.0])
        # Simulate a series of maneuvers (similar to the provided test code)
        s = get_traj(s, (5, 5), 20)
        s = get_traj(s, (-5, -5), 20)
        s = get_traj(s, (5, 5), 20)
        s = get_traj(s, (-5, -5), 20)
        s = get_traj(s, (5, 5), 20)
        
        print("Displaying test trajectory plot...")
        plt.show()
        plt.close()