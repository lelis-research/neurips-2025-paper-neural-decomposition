import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticContinuous(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCriticContinuous, self).__init__()
        obs_dim = int(np.prod(observation_space.shape))# + 3
        action_dim = int(np.prod(action_space.shape))

        # # FOR THE POINTMAZE ENVIRONMENT
        # # policy: outputs logits over actions
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
            # nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
        # # FOR THE ANTMAZE ENVIRONMENT
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(obs_dim, 256)),
        #     nn.LayerNorm(256),
        #     nn.Tanh(),

        #     layer_init(nn.Linear(256, 256)),
        #     nn.LayerNorm(256),
        #     nn.Tanh(),

        #     layer_init(nn.Linear(256, action_dim), std=0.01),
        # )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(obs_dim, 256)),
        #     nn.LayerNorm(256),
        #     nn.Tanh(),

        #     layer_init(nn.Linear(256, 256)),
        #     nn.LayerNorm(256),
        #     nn.Tanh(),

        #     layer_init(nn.Linear(256, 1), std=1.0),
        # )

        # FOR THE CAR ENVIRONMENT
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(obs_dim, 128)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(128, 128)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(128, action_dim), std=0.01),
        # )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(obs_dim, 128)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(128, 128)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(128, 1), std=1.0)
        # )


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
    
    
    def analyze_weights(self, actor_mean=None, topk: int = 100):
        """
        Prints out:
          1) All parameter names and shapes,
          2) A per-input-feature importance score for the first actor_linear layer,
             sorted by descending importance.
        """
        if actor_mean is None:
            actor_mean = self.actor_mean
        # 1) list all params
        print("=== Parameter shapes ===")
        for name, p in self.named_parameters():
            print(f"{name:40s}  {tuple(p.shape)}")
        print()

        # 2) feature importance on first actor layer
        first_lin = actor_mean[0]  # nn.Linear(obs_dim → 128)
        W = first_lin.weight.data      # shape [128, obs_dim]
        imp = W.abs().sum(dim=0)       # [obs_dim]

        # sort features
        vals, idxs = torch.sort(imp, descending=True)
        print(f"=== Top {topk} input features by abs‐weight sum ===")
        for rank, (i, v) in enumerate(zip(idxs[:topk], vals[:topk]), 1):
            print(f"{rank:2d}. feature[{i.item():3d}] → importance {v.item():.4f}")