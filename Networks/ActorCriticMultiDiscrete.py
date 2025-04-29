import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticMultiDiscrete(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        
        # shared feature extractor for both single- and multi-discrete
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # set up policy heads
        if isinstance(action_space, Discrete):
            # single discrete
            self.is_multi = False
            n = action_space.n
            self.actor_head = layer_init(nn.Linear(64, n), std=0.01)
        elif isinstance(action_space, MultiDiscrete):
            # multi-discrete: e.g. [10,10]
            self.is_multi = True
            self.nvec = action_space.nvec
            self.actor_head1 = layer_init(nn.Linear(64, self.nvec[0]), std=0.01)
            self.actor_head2 = layer_init(nn.Linear(64, self.nvec[1]), std=0.01)
        else:
            raise ValueError("Unsupported action space")

        # value function head (same for both)
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
        """
        Returns:
          action:    tensor shape (batch, ) for Discrete,
                     or (batch,2) for MultiDiscrete
          log_prob:  tensor shape (batch,)
          entropy:   tensor shape (batch,) or scalar
        """
        h = self.backbone(x)

        if not self.is_multi:
            # single Discrete
            logits = self.actor_head(h)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample() if not greedy else logits.argmax(dim=-1)
            logp = dist.log_prob(action)
            ent  = dist.entropy()
            return action, logp, ent

        # MultiDiscrete case
        logits1 = self.actor_head1(h)
        logits2 = self.actor_head2(h)
        dist1 = Categorical(logits=logits1)
        dist2 = Categorical(logits=logits2)

        if action is None:
            a1 = dist1.sample() if not greedy else logits1.argmax(dim=-1)
            a2 = dist2.sample() if not greedy else logits2.argmax(dim=-1)
        else:
            # assume action is a tensor of shape (batch,2)
            a1 = action[..., 0]
            a2 = action[..., 1]

        action = torch.stack([a1, a2], dim=-1)           # (batch,2)
        logp   = dist1.log_prob(a1) + dist2.log_prob(a2) # (batch,)
        ent    = dist1.entropy() + dist2.entropy()       # (batch,)
        return action, logp, ent