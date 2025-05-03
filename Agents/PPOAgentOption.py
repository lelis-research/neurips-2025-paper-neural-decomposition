from Networks.ActorCriticDiscrete import ActorCriticDiscrete

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical



class PPOAgentOption:
    def __init__(self, observation_space, options_list, **kwargs):
        """
        Hierarchical PPO: high‐level policy over discrete options.
        options_list: List[Option], where each Option has .max_len and .act(obs)
        """
        # Return Calculation Params 
        self.gamma    = kwargs.get("gamma", 0.99)
        self.lamda    = kwargs.get("lamda", 0.95)
        self.epochs   = kwargs.get("epochs", 10)
        
        # Batch Update Params
        self.total_steps   = kwargs.get("total_steps", 0)
        self.rollout_steps = kwargs.get("rollout_steps", 2048)
        self.num_minibatches = kwargs.get("num_minibatches", 32)
        self.minibatch_size  = self.rollout_steps // self.num_minibatches
        self.total_updates   = (self.total_steps // self.rollout_steps) + 1
        
        # Step Size Params
        self.flag_anneal_step_size = kwargs.get("flag_anneal_step_size", True)
        self.step_size   = kwargs.get("step_size", 3e-4)

        # Loss Calculation Params
        self.entropy_coef= kwargs.get("entropy_coef", 0.0)
        self.critic_coef = kwargs.get("critic_coef", 0.5)
        self.clip_ratio  = kwargs.get("clip_ratio", 0.2)
        self.flag_clip_critic_loss = kwargs.get("flag_clip_vloss", True)
        self.flag_norm_adv = kwargs.get("flag_norm_adv", True)
        self.max_grad_norm= kwargs.get("max_grad_norm", 0.5)

        self.observation_space = observation_space
        self.options           = options_list
        
        self.option_space      = gym.spaces.Discrete(len(options_list))
        self.device            = kwargs["device"]

        # high‐level actor‐critic over options
        self.actor_critic = ActorCriticDiscrete(observation_space, self.option_space).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size, eps=1e-5)

        # rollout buffer
        self.memory = []
        self.update_counter = 0
        self.ep_counter     = 0

        # option‐execution state
        self.current_option     = None   # index of chosen option
        self.option_step_count  = 0      # how many primitives we've executed under it

              
    def act(self, observation, greedy=False):
        """
        1) If no option is active, or we've reached its max_len,
           sample a new option from the high‐level policy.
        2) Execute that option for up to opt.max_len steps.
        """
        state = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)

        # pick new option if needed
        if (self.current_option is None or
            self.option_step_count >= self.options[self.current_option].max_len):
            # sample high‐level policy
            with torch.no_grad():
                option_idx_tensor, log_prob, _ = self.actor_critic.get_action(state, greedy=greedy)
            
            self.current_option    = option_idx_tensor.item()
            self.last_log_prob     = log_prob
            self.prev_state        = state
            self.prev_option       = option_idx_tensor.squeeze(0)
            self.option_step_count = 0

        # execute the chosen option's primitive policy
        action = self.options[self.current_option].act(observation)  # returns a torch.Tensor

        # increment the counter
        self.option_step_count += 1

        return action.detach().cpu().numpy()
    
    def update(self, next_observation, reward, terminated, truncated):
        """
        Store the transition *at every primitive step*,
        but the stored 'option' and 'log_prob' come from when it was sampled.
        """
        next_state = torch.tensor(next_observation, device=self.device, dtype=torch.float32).unsqueeze(0)

        self.memory.append({
            "state":      self.prev_state,     # (1, obs_dim)
            "option":     self.prev_option,    # scalar tensor
            "log_prob":   self.last_log_prob,  # scalar tensor
            
            "next_state": next_state,          # (1, obs_dim)
            "reward":     reward,              # float
            "done":       terminated or truncated,
        })

        if terminated or truncated:
            self.ep_counter += 1

        # once we've collected enough *primitives*, do PPO on options
        if len(self.memory) >= self.rollout_steps:
            self.ppo_update()
            self.memory = []

    def ppo_update(self):
        """
        Exactly the same as your discrete‐action PPO,
        but the 'action' is now the option index.
        """
        self.update_counter += 1
        # (1) lr annealing
        if self.flag_anneal_step_size:
            frac = 1.0 - (self.update_counter - 1) / self.total_updates
            self.optimizer.param_groups[0]["lr"] = frac * self.step_size

        # (2) unpack buffer
        states    = torch.cat([t["state"] for t in self.memory],   0).to(self.device)
        options   = torch.stack([t["option"] for t in self.memory]).to(self.device)
        old_lps   = torch.stack([t["log_prob"] for t in self.memory]).squeeze().to(self.device)
        next_s    = torch.cat([t["next_state"] for t in self.memory], 0).to(self.device)
        rewards   = [t["reward"] for t in self.memory]
        dones     = [t["done"]   for t in self.memory]

        with torch.no_grad():
            vals      = self.actor_critic.get_value(states).squeeze()      # (T,)
            next_vals = self.actor_critic.get_value(next_s).squeeze()      # (T,)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns    = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # GAE
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = (
                rewards[t]
                + self.gamma * next_vals[t].item() * mask
                - vals[t].item()
            )
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + vals[t].item()

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns    = torch.tensor(returns,    dtype=torch.float32, device=self.device)
        if self.flag_norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # (3) PPO epochs
        idxs = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.minibatch_size):
                mb = idxs[start : start + self.minibatch_size]

                bs = states[mb]
                bo = options[mb]
                bl = old_lps[mb]
                br = returns[mb]
                ba = advantages[mb]
                bv = vals[mb]

                # new log‐prob & entropy over options
                _, new_lps, ent = self.actor_critic.get_action(bs, bo)
                ratio = torch.exp(new_lps - bl)

                # actor loss
                s1 = -ba * ratio
                s2 = -ba * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = torch.max(s1, s2).mean()

                # critic loss
                new_vals = self.actor_critic.get_value(bs).squeeze()
                if self.flag_clip_critic_loss:
                    unclipped = (br - new_vals).pow(2)
                    clipped   = (br - (bv + torch.clamp(
                                  new_vals - bv,
                                  -self.clip_ratio,
                                  self.clip_ratio
                              ))).pow(2)
                    critic_loss = 0.5 * torch.max(clipped, unclipped).mean()
                else:
                    critic_loss = 0.5 * (br - new_vals).pow(2).mean()

                ent_bonus = ent.mean()
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * ent_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        
    def save(self, file_path):
        ckpt = {
            "gamma": self.gamma,
            "lamda": self.lamda,

            "epochs": self.epochs,
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,

            "step_size": self.step_size,
            "entropy_coef": self.entropy_coef,
            "critic_coef": self.critic_coef,
            "clip_ratio": self.clip_ratio,
            "flag_clip_vloss": self.flag_clip_critic_loss,
            "flag_norm_adv": self.flag_norm_adv,
            "max_grad_norm": self.max_grad_norm,
            
            "observation_space": self.observation_space,
            "option_space":      self.option_space,
            "actor_critic":      self.actor_critic.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
        }
        torch.save(ckpt, file_path)
        print(f"PPOAgentOption saved to {file_path}")

    @classmethod
    def load(cls, file_path, options_list):
        ckpt = torch.load(file_path)
        init_args = {
            k: v for k, v in ckpt.items()
            if k not in ("actor_critic", "optimizer")
        }
        obs_space = init_args.pop("observation_space")
        agent = cls(obs_space, options_list, **init_args)
        agent.actor_critic.load_state_dict(ckpt["actor_critic"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"PPOAgentOption loaded from {file_path}")
        return agent
        