import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import numpy as np
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Actor, self).__init__()
        obs_dim = int(np.prod(observation_space.shape)) # + 3
        action_dim = int(np.prod(action_space.shape))

        # policy: outputs logits over actions
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
            # nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, x, action=None, greedy=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
    
        if action is None:
            if greedy:
                action = action_mean
            else:
                action = probs.rsample()
                
        return action

class Actor_Discrete(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Actor_Discrete, self).__init__()
        obs_dim = int(np.prod(observation_space.shape))  + 3
        n_actions = action_space.n

        # policy: outputs logits over actions
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
            # nn.Tanh(),
        )
    
    def get_action(self, x, action=None, greedy=False):
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
                
        return action, logits

