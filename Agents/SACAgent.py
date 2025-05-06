import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

from Agents.mode_policy import update_mode, get_mode_action
# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.device = device

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) 
        # Convert to arrays, detaching tensors if needed
        states = [transition['state'] for transition in batch]
        next_states = [transition['next_state'] for transition in batch]
        actions = [transition['action'] for transition in batch]
        rewards = [transition['reward'] for transition in batch]
        dones = [transition['done'] for transition in batch]



        states = torch.cat(states, dim=0).to(self.device).float().detach()          # shape (T, obs_dim)
        actions = torch.cat(actions, dim=0).to(self.device).float().detach()          # shape (T, act_dim)
        next_states = torch.cat(next_states, dim=0).to(self.device).float().detach() # shape (T, obs_dim)
        rewards = torch.from_numpy(np.array(rewards)).unsqueeze(1).to(self.device).float().detach()  # shape (T,)
        dones = torch.from_numpy(np.array(dones)).unsqueeze(1).to(self.device).float().detach()  # shape (T,)

        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)

# Q-network approximator
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.net.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Gaussian policy with tanh-squashing and log-prob correction
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net.apply(weights_init_)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, state):
        x = self.net(state)
        mu = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # log prob correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def sample_deterministic(self, state):
        mu, _ = self(state)
        return torch.tanh(mu)

# Soft Actor-Critic Agent
class SACAgent:
    def __init__(self, observation_space, action_space, device,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 hidden_dim=256, buffer_capacity=100000, batch_size=256,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 automatic_entropy_tuning=True):
        # Spaces & device
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Dimensions
        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = int(np.prod(observation_space.shape))
        else:
            raise NotImplementedError("Unsupported observation space")
        if isinstance(action_space, gym.spaces.Box):
            act_dim = int(np.prod(action_space.shape))
        else:
            raise NotImplementedError("SAC only supports continuous action spaces (Box)")

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, device)

        # Networks
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.mode = 0
        self.ep_counter = 0

    def act(self, observation, greedy=False):
        self.mode = update_mode(self.mode, observation)
        true_action, action_discrete = get_mode_action(self.mode)
        
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        if self.ep_counter <= 5:
            action = torch.tensor(true_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            if greedy:
                action = self.policy.sample_deterministic(state)
            else:
                action, _ = self.policy.sample(state)
        

        self.prev_state = state
        self.prev_action = action
        return action.cpu().detach().numpy()[0]

    def update(self, next_observation, reward, terminated, truncated):
        next_state = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0)
        transition = {
            'state': self.prev_state,    # tensor shape (1, obs_dim)
            'action': self.prev_action,  # tensor shape (1, act_dim)
            'next_state': next_state, # tensor shape (1, obs_dim)
            'reward': reward, # scalar
            'done': terminated # boolean
        }
        
        self.replay_buffer.push(transition)

        if terminated or truncated:
            self.mode = 0
            self.ep_counter += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        
        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        q2_loss = nn.MSELoss()(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy update
        action_new, log_prob_new = self.policy.sample(states)
        q1_new = self.q1(states, action_new)
        q2_new = self.q2(states, action_new)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob_new - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Soft updates
        for param, target in zip(self.q1.parameters(), self.q1_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for param, target in zip(self.q2.parameters(), self.q2_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

    def save(self, filepath):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'alpha': self.alpha
        }, filepath)

    def load(self, filepath):
        ckpt = torch.load(filepath)
        self.policy.load_state_dict(ckpt['policy'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.q1_target.load_state_dict(ckpt['q1_target'])
        self.q2_target.load_state_dict(ckpt['q2_target'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        self.q1_optimizer.load_state_dict(ckpt['q1_optimizer'])
        self.q2_optimizer.load_state_dict(ckpt['q2_optimizer'])
        self.alpha = ckpt['alpha']
