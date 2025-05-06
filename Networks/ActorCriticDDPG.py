import torch.nn as nn
import torch

# Actor network mirroring Keras design
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, upper_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )
        self.upper_bound = upper_bound

    def forward(self, x):
        return self.net(x) * self.upper_bound

# Critic network mirroring Keras design
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(act_dim, 256), nn.ReLU()
        )
        self.joint_net = nn.Sequential(
            nn.Linear(256 * 2, 256 * 2), nn.ReLU(),
            nn.Linear(256 * 2, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        s = self.state_net(state)
        a = self.action_net(action)
        x = torch.cat([s, a], dim=-1)
        return self.joint_net(x)