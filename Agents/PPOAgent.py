from Networks.ActorCriticDiscrete import ActorCriticDiscrete
from Networks.ActorCriticContinuous import ActorCriticContinuous
from Networks.ActorCriticMultiDiscrete import ActorCriticMultiDiscrete
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

def discrete_to_continuous(action):
    if action == 0:
        action = np.array([-1., 0.])
    elif action == 1:
        action = np.array([1., 5.])
    elif action == 2:
        action = np.array([1., -3.])
    else:
        raise ValueError("Invalid action")  
    return action

def multidiscrete_to_continuous(action):
    a = np.asarray(action, dtype=np.int32)
    if a.shape[-1] != 2:
        raise ValueError(f"Expected action of length 2, got shape {a.shape}")
    
    low, high = -5.0, 5.0
    bins = 10
    # step so that 0→-5.0 and 9→+5.0, evenly spaced
    step = (high - low) / (bins - 1)
    
    cont = low + a * step
    return cont

class PPOAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        # Return Calculation Params 
        self.gamma = kwargs.get("gamma", 0.99)
        self.lamda = kwargs.get("lamda", 0.95)        

        # Batch Update Params
        self.epochs = kwargs.get("epochs", 10)
        self.total_steps = kwargs.get("total_steps")
        self.rollout_steps = kwargs.get("rollout_steps", 2048)
        self.num_minibatches = kwargs.get("num_minibatches", 32)
        self.minibatch_size = self.rollout_steps // self.num_minibatches
        self.total_updates = self.total_steps // self.rollout_steps + 1
        
        # Step Size Params
        self.flag_anneal_step_size = kwargs.get("flag_anneal_step_size", True)
        self.step_size = kwargs.get("step_size", 3e-4)
        
        # Loss Calculation Params
        self.entropy_coef = kwargs.get("entropy_coef", 0.0) 
        self.critic_coef = kwargs.get("critic_coef", 0.5)
        self.clip_ratio = kwargs.get("clip_ratio", 0.2) # Loss Clipping Ratio
        self.flag_clip_critic_loss = kwargs.get("flag_clip_vloss", True) # Clipping Critic Loss
        self.flag_norm_adv = kwargs.get("flag_norm_adv", True) # Normalizing Advantages
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5) # Clipping Gradients

        # self.kl_target = kwargs.get("kl_target", 1.0)

        # action_space = gym.spaces.Discrete(3)
        # action_space = gym.spaces.MultiDiscrete([10, 10])

        self.observation_space = observation_space
        self.action_space = action_space
        self.device = kwargs['device']

        if isinstance(action_space, gym.spaces.Discrete):
            self.actor_critic = ActorCriticDiscrete(observation_space, action_space).to(self.device)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.actor_critic = ActorCriticMultiDiscrete(observation_space, action_space).to(self.device)
        else:
            self.actor_critic = ActorCriticContinuous(observation_space, action_space).to(self.device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size, eps=1e-5)

        # Buffer to store transitions until update
        self.memory = []

        self.update_counter = 0
        
        self.ep_counter = 0
        self.exploration_lst = {}
              
    def act(self, observation, greedy=False):
        """
        Given an observation, sample an action according to the current policy.
        Stores the observation for the later update step.
        """
        state = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor_critic.get_action(state, greedy=greedy)

        self.prev_state = state
        self.last_log_prob = log_prob
        self.prev_action = action
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            return discrete_to_continuous(action.squeeze(0).detach().cpu().numpy())
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return multidiscrete_to_continuous(action.squeeze(0).detach().cpu().numpy())
        else:
            return action.squeeze(0).detach().cpu().numpy()
        
    
    
    def update(self, next_observation, reward, terminated, truncated):
        """
        Called at each step: store the transition and, if the buffer is full,
        perform a PPO update using a batch of transitions.
        """
        obs_tup = tuple(next_observation.tolist())
        if obs_tup not in self.exploration_lst:
            self.exploration_lst[obs_tup] = 1
            # reward += 1.0
        else:
            self.exploration_lst[obs_tup] += 1

        next_state = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0)
        self.memory.append({
            'state': self.prev_state,    # tensor shape (1, obs_dim)
            'action': self.prev_action,  # tensor shape (1, act_dim)
            'log_prob': self.last_log_prob,

            'next_state': next_state, # tensor shape (1, obs_dim)
            'reward': reward, # scalar
            'done': terminated # boolean
        })
        if terminated or truncated:
            self.ep_counter += 1

        if len(self.memory) >= self.rollout_steps:
            self.ppo_update()
            self.memory = []  # Clear the buffer after update


    def ppo_update(self):
        """
        Compute advantage estimates using GAE, and perform PPO updates over the collected batch.
        """
        self.update_counter += 1

        # Update the step size
        if self.flag_anneal_step_size:
            frac = 1.0 - (self.update_counter - 1.0) / self.total_updates
            self.optimizer.param_groups[0]["lr"] = frac * self.step_size
            
        # Convert memory to lists (and later tensors).
        states = [transition['state'] for transition in self.memory]
        actions = [transition['action'] for transition in self.memory]
        log_probs = [transition['log_prob'] for transition in self.memory]

        next_states = [transition['next_state'] for transition in self.memory]
        rewards = [transition['reward'] for transition in self.memory]
        dones = [transition['done'] for transition in self.memory]

        states = torch.cat(states, dim=0).to(self.device)          # shape (T, obs_dim)
        actions = torch.cat(actions, dim=0).to(self.device)          # shape (T, act_dim)
        old_log_probs = torch.stack(log_probs).squeeze().to(self.device)  # shape (T,)
        next_states = torch.cat(next_states, dim=0).to(self.device) # shape (T, obs_dim)
     
        with torch.no_grad():
            prev_state_values = self.actor_critic.get_value(states) # shape (T, 1)
            next_state_values = self.actor_critic.get_value(next_states) # shape (T, 1)


        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # Compute advantages backward using GAE.
        for t in reversed(range(T)):
            mask = 1.0 if not dones[t] else 0.0
            delta = rewards[t] + self.gamma * next_state_values[t].item() * mask - prev_state_values[t].item()
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae
            returns[t] = gae + prev_state_values[t].item()

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if self.flag_norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # PPO update over multiple epochs.
        indices = np.arange(T)
        for epoch in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                batch_states = states[mb_idx] # shape (B, obs_dim)
                batch_actions = actions[mb_idx] # shape (B, action_dim)
                batch_old_log_probs = old_log_probs[mb_idx] # shape (B, )
                batch_returns = returns[mb_idx] # shape (B, )
                batch_advantages = advantages[mb_idx] # shape (B, )
                batch_values = prev_state_values[mb_idx].squeeze() # shape (B, )

                # Recompute log probabilities and state values for current policy.
                _, batch_new_log_probs, entropy = self.actor_critic.get_action(batch_states, batch_actions)
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective.
                surrogate1 = -batch_advantages * ratio
                surrogate2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                actor_loss = torch.max(surrogate1, surrogate2).mean()

                # Critic loss.
                batch_new_values = self.actor_critic.get_value(batch_states).squeeze()
                if self.flag_clip_critic_loss:
                    critic_loss_unclipped = (batch_returns - batch_new_values).pow(2)
                    value_clipped = batch_values + torch.clamp(batch_new_values - batch_values, 
                                                               -self.clip_ratio, self.clip_ratio)
                    critic_loss_clipped = (batch_returns - value_clipped).pow(2)
                    critic_loss = 0.5 * torch.max(critic_loss_clipped, critic_loss_unclipped).mean()
                else:
                    critic_loss = 0.5 * (batch_returns - batch_new_values).pow(2).mean()

                entropy_bonus = entropy.mean()

                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_bonus
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        
    def save(self, file_path):
        checkpoint = {
            "gamma": self.gamma,
            "lamda": self.lamda,
            
            "epochs": self.epochs,
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "num_minibatches": self.num_minibatches,
            "minibatch_size": self.minibatch_size,

            "flag_anneal_step_size": self.flag_anneal_step_size,
            "step_size": self.step_size,

            "entropy_coef": self.entropy_coef,
            "critic_coef": self.critic_coef,
            "clip_ratio": self.clip_ratio,
            "flag_clip_critic_loss": self.flag_clip_critic_loss,
            "flag_norm_adv": self.flag_norm_adv,
            "max_grad_norm": self.max_grad_norm,

            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "device": self.device,

            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)
        print(f"Agent saved to {file_path}")
    
    @classmethod
    def load(cls, file_path):
        checkpoint = torch.load(file_path, weights_only=False) 
        init_kwargs = {
            k: v
            for k, v in checkpoint.items()
            if k not in ("actor_critic", "optimizer")
        }

        observation_space = init_kwargs.pop("observation_space")
        action_space = init_kwargs.pop("action_space")

        agent = cls(observation_space, action_space, **init_kwargs)
        
        agent.actor_critic.load_state_dict(checkpoint["actor_critic"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Agent loaded from {file_path}")
        return agent
        