import torch
import torch.nn as nn
import numpy as np

# Import the same initialization utility used in ActorCriticContinuous
from Networks.ActorCriticContinuous import layer_init

class QNetwork(nn.Module):
    """
    Feedforward Q-network mirroring the structure of ActorCriticContinuous.

    Args:
        observation_space (gym.Space): Environment observation space.
        action_space (gym.Space): Environment action space (Discrete).
    """
    def __init__(self, observation_space, action_dim):
        super(QNetwork, self).__init__()
        obs_dim = int(np.prod(observation_space.shape))

        self.q_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each action given state batch.

        Args:
            x (torch.Tensor): State tensor of shape (batch, *obs_shape).

        Returns:
            torch.Tensor: Q-values tensor of shape (batch, action_dim).
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.q_net(x)