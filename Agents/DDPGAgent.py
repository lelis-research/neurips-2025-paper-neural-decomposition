from Networks.ActorCriticDDPG import Actor, Critic

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import random

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.stack(actions)),
            torch.FloatTensor(np.stack(rewards)).unsqueeze(1),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(np.stack(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# Ornstein-Uhlenbeck noise for exploration
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.size)
        self.state = self.state + dx
        return self.state

# DDPG agent with PPO-like structure
class DDPGAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        # Initialize hyperparameters
        self.initialize_params(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = kwargs['device']

        # Dimensions
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))

        # Actor and target
        self.actor = Actor(obs_dim, act_dim, self.upper_bound).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic and target
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.noise = OUNoise(size=(act_dim,), mu=0.0, theta=self.ou_theta, sigma=self.ou_sigma)
        self.epsilon = self.epsilon_start

        # State placeholder
        self.prev_state = None
        self.prev_action = None

        # Counters
        self.update_counter = 0

    def initialize_params(self, **kwargs):
        # Learning rates
        self.actor_lr = kwargs.get('actor_lr', 1e-3)
        self.critic_lr = kwargs.get('critic_lr', 1e-3)
        # Discount and polyak
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        # Replay buffer
        self.buffer_capacity = int(kwargs.get('buf_size', 100_000))
        self.batch_size = kwargs.get('batch_size', 64)
        # Network sizes
        # Action bounds
        self.lower_bound, self.upper_bound = (-5.0, 5.0)
        # Exploration noise
        self.noise_phi = kwargs.get('noise_phi', 0.2)
        self.ou_theta = kwargs.get('ou_theta', 0.15)
        self.ou_sigma = kwargs.get('ou_sigma', 0.2)
        # Epsilon schedule
        self.epsilon_start = 1.0
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = 1.0 / kwargs.get('EXPLORE', 100_000.0)

    def act(self, observation, greedy=False):
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_np = self.actor(state).cpu().numpy()[0]
        # Store for update
        self.prev_state = state
        self.prev_action = action_np
        # Add exploration noise if not greedy
        if not greedy:
            noise = self.noise.sample() * self.epsilon
            action_np = np.clip(action_np + noise, self.lower_bound, self.upper_bound)
        return action_np

    def update(self, next_observation, reward, terminated, truncated):
        # Store transition using previous state/action
        if self.prev_state is not None and self.prev_action is not None:
            state_np = self.prev_state.cpu().numpy()[0]
            self.replay_buffer.push(state_np, self.prev_action, reward, next_observation, float(terminated))
        # Prepare next state for the next step
        self.prev_state = torch.FloatTensor(next_observation).to(self.device).unsqueeze(0)
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        # Learn if enough data
        if len(self.replay_buffer) >= self.batch_size:
            self.learn()

    def learn(self):
        self.update_counter += 1
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * q_next
        q_current = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update targets
        for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

    def save(self, file_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }, file_path)

    @classmethod
    def load(cls, file_path):
        ckpt = torch.load(file_path)
        raise NotImplementedError("Instantiate via __init__ and load weights manually.")
