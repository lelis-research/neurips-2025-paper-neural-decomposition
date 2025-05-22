import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

from Networks.QNetwork import QNetwork
import numpy as np
from typing import Tuple

class DQNAgent:
    """
    Deep Q-Network (DQN) agent following PPOAgent structure:
    - __init__, initialize_params, act, update, and save/load methods.
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        print("Initialize DQN Agent")
        self.initialize_params(**kwargs)
        
        low, high = (-5, -5), (5, 5)  # each of shape (2,)
        xs = np.linspace(low[0], high[0], self.action_res)
        ys = np.linspace(low[1], high[1], self.action_res)
        # Cartesian product → (res*res, 2)
        self._action_grid = np.array([[x,y] for x in xs for y in ys], dtype=np.float32)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = kwargs.get('device', 'cpu')
        self.action_dim = len(self._action_grid)
        
        # Q-network and target network
        self.q_network = QNetwork(observation_space, self.action_dim).to(self.device)
        self.target_network = QNetwork(observation_space, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.step_size)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = deque(maxlen=self.replay_buffer_cap)
        self.update_counter = 0
        self.ep_counter = 0

    def initialize_params(self, **kwargs):
        # Core hyperparameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.step_size = kwargs.get('step_size', 0.0001)
        self.batch_size = kwargs.get('batch_size', 64)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.replay_buffer_cap = kwargs.get('replay_buffer_cap', 2_000_000)
        self.action_res = kwargs.get('action_res', 3)

    def act(self, observation: np.ndarray, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        """
        if not greedy and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim-1)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = q_values.argmax(dim=1).item()

        # Store for next update
        self.prev_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.prev_action = action
        return self._action_grid[action]

    def update(self, next_observation: np.ndarray, reward: float,
               terminated: bool, truncated: bool):
        """
        Store transition and learn if enough samples.
        """
        next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        # append experience
        self.memory.append((self.prev_state, self.prev_action, reward, next_state, terminated))

        if terminated or truncated:
            self.ep_counter += 1

        if len(self.memory) >= self.batch_size:
            self.dqn_update()

    def dqn_update(self):
        """
        Sample batch and perform a DQN update.
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # current Q-values
        q_vals = self.q_network(states).gather(1, actions)

        # target Q-values
        with torch.no_grad():
            max_next = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1 - dones) * max_next

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, file_path: str):
        """
        Save model weights and hyperparameters.
        """
        checkpoint = {
            'gamma': self.gamma,
            'step_size': self.step_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'epsilon': self.epsilon,
            'replay_buffer_cap': self.replay_buffer_cap,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'device': self.device,
            'action_res':self.action_res
        }
        torch.save(checkpoint, file_path)
        print(f"DQNAgent saved to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """
        Load a DQNAgent from disk.
        """
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        init_kwargs = {k: checkpoint[k] for k in [
            'gamma','step_size','batch_size',
            'target_update_freq','epsilon','replay_buffer_cap','device', 'action_res'
        ]}
        agent = cls(
            checkpoint['observation_space'],
            checkpoint['action_space'],
            **init_kwargs
        )
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        agent.target_network.load_state_dict(checkpoint['target_network_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"DQNAgent loaded from {file_path}")
        return agent