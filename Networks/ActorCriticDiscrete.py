import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticDiscrete(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticDiscrete, self).__init__()
        obs_dim = int(np.prod(observation_space.shape))
        n_actions = action_space.n  # assumes a gym.spaces.Discrete

        # policy: outputs logits over actions
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

        # value function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None, greedy=False):
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
                
        return action, dist.log_prob(action), dist.entropy()