import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque, namedtuple

from Networks.QNetwork import QNetwork

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class NStepDQNAgent:
    """
    DQN agent with N-step returns and epsilon decay.
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        print("Initialize N-step DQN Agent")
        self.initialize_params(**kwargs)

        # Action grid (for discretized action space)
        low, high = (-5, -5), (5, 5)
        xs = np.linspace(low[0], high[0], self.action_res)
        ys = np.linspace(low[1], high[1], self.action_res)
        self._action_grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
        self.action_dim = len(self._action_grid)

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = kwargs.get('device', 'cpu')

        # Primary and target networks
        self.q_network = QNetwork(observation_space, self.action_dim).to(self.device)
        self.target_network = QNetwork(observation_space, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.step_size)
        self.loss_fn = nn.MSELoss()

        # Replay and N-step buffers
        self.memory = deque(maxlen=self.replay_buffer_cap)
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.update_counter = 0
        self.total_steps = 0

    def initialize_params(self, **kwargs):
        # Core hyperparameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.step_size = kwargs.get('step_size', 0.0001)
        self.batch_size = kwargs.get('batch_size', 64)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        # Epsilon decay
        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.001)
        self.epsilon_decay_steps = kwargs.get('epsilon_decay_steps', 500_000)
        self.epsilon = self.epsilon_start
        # Replay and N-step
        self.replay_buffer_cap = kwargs.get('replay_buffer_cap', 2_000_000)
        self.action_res = kwargs.get('action_res', 3)
        self.n_step = kwargs.get('n_step', 7)

    def act(self, observation: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Epsilon-greedy action selection with linear decay.
        """
        # Decay epsilon
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

        if not greedy and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            action_idx = q_values.argmax(dim=1).item()

        # store last
        self.prev_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.prev_action = action_idx
        self.total_steps += 1
        return self._action_grid[action_idx]

    def update(self, next_observation: np.ndarray, reward: float,
               terminated: bool, truncated: bool):
        """
        Store transitions and trigger learning.
        """
        next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.n_step_buffer.append(Transition(self.prev_state, self.prev_action,
                                             reward, next_state, terminated))

        # Once N steps collected, push aggregated transition
        if len(self.n_step_buffer) == self.n_step:
            R, state_0, action_0, next_s_n, done_n = self._get_n_step_info()
            self.memory.append((state_0, action_0, R, next_s_n, done_n))

        # Flush at episode end
        if terminated or truncated:
            while self.n_step_buffer:
                R, state_0, action_0, next_s_n, done_n = self._get_n_step_info()
                self.memory.append((state_0, action_0, R, next_s_n, done_n))

        if len(self.memory) >= self.batch_size:
            self._learn()

    def _get_n_step_info(self):
        """
        Compute return over N steps and pop oldest.
        """
        R = 0.0
        for idx, trans in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * trans.reward
            if trans.done:
                break
        state_0  = self.n_step_buffer[0].state
        action_0 = self.n_step_buffer[0].action
        last     = self.n_step_buffer[-1]
        next_state_n = last.next_state
        done_n       = last.done
        self.n_step_buffer.popleft()
        return R, state_0, action_0, next_state_n, done_n

    def _learn(self):
        """
        Sample batch and perform DQN update.
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states     = torch.cat(states)
        actions    = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards    = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states= torch.cat(next_states)
        dones      = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q
        q_vals = self.q_network(states).gather(1, actions)
        # target Q
        with torch.no_grad():
            max_next = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (self.gamma ** self.n_step) * (1 - dones) * max_next

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, file_path: str):
        """
        Save agent state and parameters.
        """
        checkpoint = {
            'gamma': self.gamma,
            'step_size': self.step_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'replay_buffer_cap': self.replay_buffer_cap,
            'action_res': self.action_res,
            'n_step': self.n_step,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'device': self.device
        }
        torch.save(checkpoint, file_path)
        print(f"NstepDQNAgent saved to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """
        Load an NstepDQNAgent from disk.
        """
        checkpoint = torch.load(file_path, map_location='cpu')
        init_kwargs = {
            'gamma': checkpoint['gamma'],
            'step_size': checkpoint['step_size'],
            'batch_size': checkpoint['batch_size'],
            'target_update_freq': checkpoint['target_update_freq'],
            'epsilon_start': checkpoint['epsilon_start'],
            'epsilon_end': checkpoint['epsilon_end'],
            'epsilon_decay_steps': checkpoint['epsilon_decay_steps'],
            'replay_buffer_cap': checkpoint['replay_buffer_cap'],
            'action_res': checkpoint['action_res'],
            'n_step': checkpoint['n_step'],
            'device': checkpoint['device']
        }
        agent = cls(
            checkpoint['observation_space'],
            checkpoint['action_space'],
            **init_kwargs
        )
        agent.q_network.load_state_dict(checkpoint['q_network_state'])
        agent.target_network.load_state_dict(checkpoint['target_network_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"NstepDQNAgent loaded from {file_path}")
        return agent
