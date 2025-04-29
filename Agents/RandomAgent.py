import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from gymnasium.spaces import Discrete

from Networks.Actor import Actor, Actor_Discrete

def update_mode(mode, observation):
    x, y, ang, df, db = observation[0], observation[1], observation[2], observation[3], observation[4]

    if mode == 0:
        # going backward
        if db < 0.5:
            mode = 1

    elif mode == 1:
        # going forward an out
        if df < 0.5 and x > 0:
            mode = 0
        if y > 5 and x < -1:
            mode = 2
    return mode

def get_mode_action(mode):
    if mode == 0:
        action = -1, 0
        action_discrete = 0
    elif mode == 1:
        action = 1, 5
        action_discrete = 1
    elif mode == 2:
        action = 1, -3
        action_discrete = 2
    return action, action_discrete

def discrete_to_continuous(action):
    if action == 0:
        action = -1, 0
    elif action == 1:
        action = 1, 5
    elif action == 2:
        action = 1, -3
    else:
        raise ValueError("Invalid action")  
    return action


class RandomAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self.device = kwargs['device']
        self.epochs = kwargs.get("epochs", 10)
        
        self.rollout_steps = kwargs.get("rollout_steps", 2048)
        self.num_minibatches = kwargs.get("num_minibatches", 32)
        self.minibatch_size = self.rollout_steps // self.num_minibatches

        self.step_size = kwargs.get("step_size", 3e-4)


        self.observation_space = observation_space
        
        self.mode = 0
        self.memory = []
        
        self.act_teacher = False
        self.num_teacher_rollouts = 0
        self.update_student = True
        self.discrete_action = True

        if self.discrete_action:
            self.action_space = Discrete(3)
            self.actor = Actor_Discrete(observation_space, self.action_space).to(self.device)
        else:
            self.action_space = action_space
            self.actor = Actor(observation_space, self.action_space).to(self.device)

        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.step_size, eps=1e-5)
        self.teacher_rollouts_counter = 0

    def act(self, observation, greedy=False):
        self.mode = update_mode(self.mode, observation)
        action, action_discrete = get_mode_action(self.mode)
        
        # mode_onehot = np.zeros([3])
        # mode_onehot[self.mode] = 1
        # observation = np.concatenate([observation, mode_onehot])

        self.prev_state  = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.true_action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.true_action_discrete = torch.tensor(action_discrete, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.act_teacher:
            return self.true_action.squeeze(0).detach().cpu().numpy()
        else:
            self.actor.eval()
            with torch.no_grad():
                action, _ = self.actor.get_action(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0), greedy=True)
                if self.discrete_action:
                    action = discrete_to_continuous(action.squeeze(0).detach().cpu())
                else:
                    action = action.squeeze(0).detach().cpu().numpy()
                return action

    def update(self, next_observation, reward, terminated, truncated):
        self.memory.append({
            'state': self.prev_state,    # tensor shape (1, obs_dim)
            'action': self.true_action,  # tensor shape (1, act_dim)
            'action_discrete': self.true_action_discrete,  # tensor shape (1, act_dim)
        })
         
        if terminated or truncated:
            self.mode = 0

        if len(self.memory) >= self.rollout_steps and self.update_student:
            self.train_actor()
            self.memory = []  # Clear the buffer after update
            self.teacher_rollouts_counter += 1
        
        if self.teacher_rollouts_counter >= self.num_teacher_rollouts:
            self.act_teacher = False
        
    
    def train_actor(self):
        states  = torch.cat([t['state']  for t in self.memory], dim=0).to(self.device) # shape (T, obs_dim)
        actions = torch.cat([t['action'] for t in self.memory], dim=0).to(self.device) # shape (T, act_dim)
        actions_discrete = torch.cat([t['action_discrete'] for t in self.memory], dim=0).to(self.device) # shape (T, act_dim)

        # states += torch.randn_like(states) * 0.01 # add noise to states
        
        if self.discrete_action:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()

        N = states.size(0)

        self.actor.train()
        for epoch in range(1, self.epochs + 1):
            # shuffle indices
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0

            # iterate over minibatches
            for i in range(self.num_minibatches):
                idx = perm[i * self.minibatch_size : (i + 1) * self.minibatch_size]
                states_mb  = states[idx]
                actions_mb = actions[idx]
                actions_discrete_mb = actions_discrete[idx]

                # forward + loss
                pred_actions, logits = self.actor.get_action(states_mb)
                if self.discrete_action:
                    loss = loss_fn(logits.float(), actions_discrete_mb.long())
                else:
                    loss = loss_fn(pred_actions, actions_mb)

                # backward + step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / self.num_minibatches
            print(f"[Imitation] Epoch {epoch}/{self.epochs}, Avg Loss: {avg_loss:.6f}")
          


    def save(self, file_path):
        checkpoint = { 
            "epochs": self.epochs,
            "rollout_steps": self.rollout_steps,
            "num_minibatches": self.num_minibatches,
            "minibatch_size": self.minibatch_size,

            "step_size": self.step_size,

            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "device": self.device,

            "actor": self.actor.state_dict(),
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
            if k not in ("actor", "optimizer")
        }

        observation_space = init_kwargs.pop("observation_space")
        action_space = init_kwargs.pop("action_space")

        agent = cls(observation_space, action_space, **init_kwargs)
        
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Agent loaded from {file_path}")
        agent.act_teacher = False
        agent.update_student = False
        return agent

        

        
    
    