import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

from Networks.ActorCriticDiscrete import ActorCriticDiscrete
from Networks.ActorCriticContinuous import ActorCriticContinuous
from Networks.ActorCriticMultiDiscrete import ActorCriticMultiDiscrete
from Agents.PPOAgent import PPOAgent  # adjust import as needed

class ElitePPOAgent(PPOAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        print("Initialize Elite PPO Agent")
        # Extract elite-related params
        self.elite_capacity = kwargs.pop('elite_capacity', 5)
        self.elite_fraction = kwargs.pop('elite_fraction', 0.2)
        super().__init__(observation_space, action_space, **kwargs)

        # Buffer for elite episodes
        self.elite_buffer = []  # list of (return, transitions)
        self.current_episode = []
        self.current_episode_return = 0.0

    def update(self, next_observation, reward, terminated, truncated):
        # Build transition dict
        next_state = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0)
        transition = {
            'state': self.prev_state,
            'action': self.prev_action,
            'log_prob': self.last_log_prob,
            'next_state': next_state,
            'reward': reward,
            'done': terminated
        }
        # Append to rollout memory and current episode
        self.memory.append(transition)
        self.current_episode.append(transition)
        self.current_episode_return += reward

        # If episode ends: update elite buffer, reset episode
        if terminated or truncated:
            self._add_to_elite()
            self.current_episode = []
            self.current_episode_return = 0.0
            self.ep_counter += 1

        # Once rollout buffer is full, perform update
        if len(self.memory) >= self.rollout_steps:
            self.ppo_update()
            self.memory = []

    def _add_to_elite(self):
        ep_ret = self.current_episode_return
        # Deep copy transitions to freeze
        ep_trans = [t.copy() for t in self.current_episode]
        # Insert if buffer not full or return better than worst
        if len(self.elite_buffer) < self.elite_capacity:
            self.elite_buffer.append((ep_ret, ep_trans))
            self.elite_buffer.sort(key=lambda x: x[0], reverse=True)
        elif ep_ret > self.elite_buffer[-1][0]:
            self.elite_buffer[-1] = (ep_ret, ep_trans)
            self.elite_buffer.sort(key=lambda x: x[0], reverse=True)

    def ppo_update(self):
        # Increase update counter and anneal LR
        self.update_counter += 1
        if self.flag_anneal_step_size:
            frac = 1.0 - (self.update_counter - 1.0) / self.total_updates
            self.optimizer.param_groups[0]['lr'] = frac * self.step_size

        # Prepare on-policy batch (as in PPOAgent)
        batch = self.memory
        states = torch.cat([t['state'] for t in batch], dim=0).to(self.device)
        actions = torch.cat([t['action'] for t in batch], dim=0).to(self.device)
        old_log_probs = torch.stack([t['log_prob'] for t in batch]).squeeze().to(self.device)
        next_states = torch.cat([t['next_state'] for t in batch], dim=0).to(self.device)
        rewards = [t['reward'] for t in batch]
        dones = [t['done'] for t in batch]

        with torch.no_grad():
            values = self.actor_critic.get_value(states).squeeze()
            next_values = self.actor_critic.get_value(next_states).squeeze()

        # Compute GAE advantages and returns
        T = len(batch)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_values[t].item() * mask - values[t].item()
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t].item()
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.flag_norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Sample elite transitions
        elite_batch = []
        num_elite = int(T * self.elite_fraction)
        if self.elite_buffer and num_elite > 0:
            while len(elite_batch) < num_elite:
                _, ep_trans = random.choice(self.elite_buffer)
                elite_batch.extend(ep_trans)
            elite_batch = elite_batch[:num_elite]

        # If elite data exists, compute their advantages and merge
        if elite_batch:
            st_e = torch.cat([t['state'] for t in elite_batch], dim=0).to(self.device)
            ac_e = torch.cat([t['action'] for t in elite_batch], dim=0).to(self.device)
            lp_e = torch.stack([t['log_prob'] for t in elite_batch]).squeeze().to(self.device)
            ns_e = torch.cat([t['next_state'] for t in elite_batch], dim=0).to(self.device)
            rew_e = [t['reward'] for t in elite_batch]
            dn_e = [t['done'] for t in elite_batch]
            with torch.no_grad():
                val_e = self.actor_critic.get_value(st_e).squeeze()
                nv_e = self.actor_critic.get_value(ns_e).squeeze()
            Te = len(elite_batch)
            adv_e = np.zeros(Te, dtype=np.float32)
            ret_e = np.zeros(Te, dtype=np.float32)
            gae_e = 0.0
            for t in reversed(range(Te)):
                mask = 0.0 if dn_e[t] else 1.0
                delta = rew_e[t] + self.gamma * nv_e[t].item() * mask - val_e[t].item()
                gae_e = delta + self.gamma * self.lamda * mask * gae_e
                adv_e[t] = gae_e
                ret_e[t] = gae_e + val_e[t].item()
            adv_e = torch.tensor(adv_e, dtype=torch.float32, device=self.device)
            ret_e = torch.tensor(ret_e, dtype=torch.float32, device=self.device)
            if self.flag_norm_adv:
                adv_e = (adv_e - adv_e.mean()) / (adv_e.std() + 1e-8)

            # Merge on-policy and elite
            states = torch.cat((states, st_e), dim=0)
            actions = torch.cat((actions, ac_e), dim=0)
            old_log_probs = torch.cat((old_log_probs, lp_e), dim=0)
            returns = torch.cat((returns, ret_e), dim=0)
            advantages = torch.cat((advantages, adv_e), dim=0)
            T = states.shape[0]
            self.minibatch_size = max(1, T // self.num_minibatches)


        with torch.no_grad():
            values = self.actor_critic.get_value(states).squeeze()

        T = states.shape[0]
        self.minibatch_size = max(1, T // self.num_minibatches)

        # Perform PPO epochs
        idxs = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.minibatch_size):
                mb = idxs[start:start + self.minibatch_size]
                s_mb = states[mb]
                a_mb = actions[mb]
                lp_mb = old_log_probs[mb]
                r_mb = returns[mb]
                adv_mb = advantages[mb]
                _, new_lp, ent = self.actor_critic.get_action(s_mb, a_mb)
                ratio = torch.exp(new_lp - lp_mb)
                s1 = -adv_mb * ratio
                s2 = -adv_mb * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = torch.max(s1, s2).mean()
                val_new = self.actor_critic.get_value(s_mb).squeeze()
                if self.flag_clip_critic_loss:
                    uncl = (r_mb - val_new).pow(2)
                    clip_val = values[mb] + torch.clamp(val_new - values[mb], -self.clip_ratio, self.clip_ratio)
                    clp = (r_mb - clip_val).pow(2)
                    critic_loss = 0.5 * torch.max(uncl, clp).mean()
                else:
                    critic_loss = 0.5 * (r_mb - val_new).pow(2).mean()
                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
