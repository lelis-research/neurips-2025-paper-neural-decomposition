from Networks.ActorCriticDiscrete import ActorCriticDiscrete
from Networks.ActorCriticContinuous import ActorCriticContinuous
from Networks.ActorCriticMultiDiscrete import ActorCriticMultiDiscrete
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym

class A2CAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        print("Initialize A2C Agent")
        
        # Initialize hyperparameters
        self.initialize_params(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = kwargs.get("device", "cpu")
        self.hidden_size = kwargs.get("hidden_size", 64)

        # Build actor-critic model
        if isinstance(action_space, gym.spaces.Discrete):
            # Discrete actions
            self.actor_critic = ActorCriticDiscrete(observation_space, action_space, hidden_size=kwargs.get("hidden_size", 64)).to(self.device)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            # MultiDiscrete actions
            self.actor_critic = ActorCriticMultiDiscrete(observation_space, action_space).to(self.device)
        else:
            # Continuous actions
            self.actor_critic = ActorCriticContinuous(observation_space, action_space).to(self.device)

        # Separate optimizers for actor and critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size, eps=1e-5)

        # Storage for n-step rollouts
        self.memory = []  # list of dicts: {state, action, log_prob, reward, next_state, done}

    def initialize_params(self, **kwargs):
        # Return calculation
        self.gamma = kwargs.get("gamma", 0.99)
        self.lamda = kwargs.get("lamda", 0.95)
        
        # Rollout steps
        self.rollout_steps = kwargs.get("rollout_steps", 5)
        # Learning rates
        self.step_size = kwargs.get("step_size", 7e-4)

    def act(self, observation, greedy=False):
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor_critic.get_action(state, greedy=greedy)
        # Save for update
        self.prev_state = state
        self.prev_log_prob = log_prob
        self.prev_action = action


        return action.squeeze(0).cpu().numpy()

    def update(self, next_observation, reward, terminated, truncated):
        next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Store transition
        self.memory.append({
            'state': self.prev_state,
            'action': self.prev_action,
            'log_prob': self.prev_log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': terminated
        })
        # Check if rollout is ready
        if len(self.memory) >= self.rollout_steps:
            self._a2c_update()
            self.memory = []

    def _a2c_update(self):
        # Gather rollout
        states      = torch.cat([m['state']     for m in self.memory], dim=0)  # (T, …)
        actions     = torch.cat([m['action']    for m in self.memory], dim=0)  # (T,) or (T,act_dim)
        rewards     = [m['reward']              for m in self.memory]         # list length T
        next_states = torch.cat([m['next_state'] for m in self.memory], dim=0)  # (T, …)
        dones       = [float(m['done'])        for m in self.memory]          # list length T

        prev_vals = self.actor_critic.get_value(states).squeeze()    # requires_grad=True
        with torch.no_grad():
            next_vals = self.actor_critic.get_value(next_states).squeeze()  # used only as a constant
        
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns    = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_vals[t].item() * mask - prev_vals[t].item()
            gae   = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + prev_vals[t].item()
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns    = torch.tensor(returns,    dtype=torch.float32, device=self.device).unsqueeze(1)
        
        _, log_probs, _ = self.actor_critic.get_action(states, actions)
        log_probs = log_probs.squeeze()

        critic_loss = F.mse_loss(prev_vals.unsqueeze(1), returns)
        actor_loss = - (log_probs * advantages).sum()

        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
    def save(self, file_path):
        checkpoint = {
            'gamma': self.gamma,
            'rollout_steps': self.rollout_steps,
            'step_size': self.step_size,
            
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'hidden_size': self.hidden_size,
            'device': self.device
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load(cls, file_path):
        import os
        print("Current Path", os.path.curdir)
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        init_kwargs = {
            'gamma': checkpoint['gamma'],
            'rollout_steps': checkpoint['rollout_steps'],
            'step_size': checkpoint['step_size'],
            'device': checkpoint['device'],
            'hidden_size': checkpoint['hidden_size'] if 'hidden_size' in checkpoint else 64
        }
        obs_space = checkpoint['observation_space']
        act_space = checkpoint['action_space']
        agent = cls(obs_space, act_space, **init_kwargs)
        
        agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])

        return agent
