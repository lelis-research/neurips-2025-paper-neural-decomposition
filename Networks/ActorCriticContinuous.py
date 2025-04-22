import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticContinuous(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticContinuous, self).__init__()
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        # policy: outputs logits over actions
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None, greedy=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            if greedy:
                action = action_mean
            else:
                action = probs.sample()
                
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)