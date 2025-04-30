import copy
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from gymnasium.vector import SyncVectorEnv
from torch.distributions.categorical import Categorical

from environments.environments_combogrid_gym import ComboGym
from environments.environments_minigrid import MiniGridWrap

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)  # Quantize to -1 or 1
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x
    

class Trajectory:
    def __init__(self):
        self._sequence = []
        self.logits = []

    def add_pair(self, state, action, logits=None, detach=False):
        if isinstance(action, torch.Tensor):
            action = action.item()
        self._sequence.append((state, action))
        if logits is not None:
            self.logits.append(copy.deepcopy(logits.cpu().detach()) if detach else logits)

    def concat(self, other):
        self._sequence = self._sequence + copy.deepcopy(other._sequence)
        self.logits = self.logits + copy.deepcopy(other.logits)

    def slice(self, start, stop=None, n=None):
        if stop:
            end = stop
        elif n:
            end = start + n
        else:
            end = len(self._sequence)
        new = copy.deepcopy(self)
        new._sequence = self._sequence[start:end]
        new.logits = self.logits[start:end]
        return new

    def get_length(self):
        return len(self._sequence)
    
    def get_trajectory(self):
        return self._sequence
    
    def get_logits_sequence(self):
        return self.logits
    
    def get_action_sequence(self):
        return [pair[1] for pair in self._sequence]
    
    def get_state_sequence(self):
        return [pair[0] for pair in self._sequence]
    
    def __repr__(self):
        return f"Trajectory(sequence={self._sequence})"
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GruAgent(nn.Module):
    def __init__(self, envs, h_size=64, feature_extractor=False, greedy=True, quantized=0, actor_layer_size=64, critic_layer_size=64):
        super().__init__()
        if isinstance(envs, ComboGym):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs.unwrapped, MiniGridWrap):
            envs = envs.unwrapped
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, SyncVectorEnv):
            observation_space_size = envs.single_observation_space.shape[0]
            action_space_size = envs.single_action_space.n
        else:
            raise NotImplementedError
        self.input_to_actor = True
        self.hidden_size = h_size
        self.greedy = greedy
        self.quantized = quantized
        # Option attributes
        self.feature_mask = None
        self.actor_mask = None
        self.option_size = None
        self.problem_id = None
        self.environment_args = None
        self.extra_info = None
        self.observation_space_size = observation_space_size
        self.option_cache = {}

        if feature_extractor:
            self.network = nn.Sequential(
                layer_init(nn.Linear(np.array(observation_space_size).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 512)),
            )
            self.gru = nn.GRU(512, h_size, 1)
        else:
            self.network = IdentityLayer()
            self.gru = nn.GRU(observation_space_size, h_size, 1)


        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # self.actor = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(128 + envs.single_observation_space.shape[0], 1), std=1)
        if self.input_to_actor:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size + observation_space_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, action_space_size),std=np.sqrt(2)*0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size + observation_space_size, critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, 1)),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(h_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(actor_layer_size, action_space_size), std=np.sqrt(2)*0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(h_size , critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
                nn.Tanh(),
                layer_init(nn.Linear(critic_layer_size, 1)),
            )

    def get_states(self, x, gru_state, done):
        hidden = self.network(x)
        # GRU logic
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.gru(h.unsqueeze(0), (1.0 - d).view(1, -1, 1) * gru_state)
            if self.quantized == 1:
                gru_state = STEQuantize.apply(gru_state)
            new_hidden += [STEQuantize.apply(h)]
            # new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        if self.input_to_actor:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else:
            hidden, _ = self.get_states(x, gru_state, done)
            concatenated = hidden
        return self.critic(concatenated)

    def get_action_and_value(self, x, gru_state, done, action=None):
        if len(x.shape) == 1:  # If no batch dimension
            x = x[None, ...]
        if self.input_to_actor:
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = torch.cat((hidden, x), dim=1)
        else: 
            hidden, gru_state = self.get_states(x, gru_state, done)
            concatenated = hidden
        logits = self.actor(concatenated)
        probs = Categorical(logits=logits)
        if action is None:
            if self.greedy:
                action = torch.tensor([torch.argmax(logits[i]).item() for i in range(len(logits))])
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(concatenated), gru_state, logits

    #TODO: Used in other codes, will edit it to work for synced envs too
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)


    def to_option(self, mask_f, mask_a, option_size, problem_id):
        self.feature_mask = mask_f
        self.actor_mask = mask_a
        self.option_size = option_size
        self.problem_id = problem_id
        self.extra_info = {}

    def _masked_input_softmax(self, input, mask):
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        
        return (mask[0] * 0) + (mask[1] * 1) + (mask[2] * input)
    
    def _masked_hidden_state_softmax(self, hidden, mask):
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        
        return (mask[0] * -1) + (mask[1] * 1) + (mask[2] * hidden)
    
    def _masked_neuron_operation_softmax(self, logits, mask):
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        relu_out = torch.relu(logits)

        return (mask[0] * 0) + (mask[1] * logits) + (mask[2] * relu_out)


    def _masked_input(self, input, mask):
        """
        Apply a mask to the input layer.

        Parameters:
            x (torch.Tensor): Input of the model; we assume binary values. 
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass one
                                0 = pass zero
                                -1 = pass the value of the input

        Returns:
            torch.Tensor: The post-masked inputs of the model.
        """
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        
        return mask * (mask - 1) / 2 * input + mask * (mask + 1) * 1 / 2

    def _masked_neuron_operation(self, logits, mask):
        """
        Apply a mask to neuron outputs in a layer.

        Parameters:
            x (torch.Tensor): The pre-activation outputs (linear outputs) from neurons.
            mask (torch.Tensor): The mask controlling the operation, where:
                                1 = pass the linear input
                                0 = pass zero,
                                -1 = compute ReLU as usual (part of the program).

        Returns:
            torch.Tensor: The post-masked outputs of the neurons.
        """
        if mask is None or len(mask) == 0:
            raise Exception("No mask is set for the agent.")
        relu_out = torch.relu(logits)
        
        output = mask * (mask - 1) / 2 * relu_out + mask * (mask + 1) * logits / 2
        return output


    def _masked_input_forward_softmax(self, x, gru_state, mask_f=None, mask_g=None, mask_a=None):
        #TODO Fix the mask if nothing is given
        if mask_f == None: mask_f = self.feature_mask
        if mask_g == None: mask_g = self.actor_mask
        if mask_a == None: mask_a = [0, 0, 1]

        x_masked = self._masked_input_softmax(x, mask_f)
        hidden = self.network(x_masked)
        # GRU logic
        h, gru_state = self.gru(hidden, gru_state)
        if self.quantized == 1:
            gru_state = STEQuantize.apply(gru_state)
            #TODO masking of GRU hidden states
        discrete_hidden = STEQuantize.apply(h)
        
        concatenated = torch.cat((discrete_hidden, x), dim=1)
        concatenated = self._masked_input_softmax(concatenated, mask_a)

        logits = self.actor(concatenated)
        
      
        probs = Categorical(logits=logits).probs
        
        return probs, logits, gru_state

    

    def run(self, env: Union[ComboGym], length_cap=None, detach_tensors=True, verbose=False):

        trajectory = Trajectory()
        current_length = 0
        self.actor.requires_grad = False
        next_rnn_state = self.init_hidden()
        hiddens = []
        next_done = torch.zeros(1).to(device)
        o, _ = env.reset()
        
        done = False

        if verbose: print('Beginning Trajectory')
        while not done:
            o = torch.tensor(o, dtype=torch.float32)
            hiddens.append(next_rnn_state.clone())
            a, _, _, _, next_rnn_state, logits = self.get_action_and_value(o, next_rnn_state, next_done)
            trajectory.add_pair(copy.deepcopy(env), a.item(), logits, detach=detach_tensors)

            if verbose:
                print(env, a)

            next_o, _, terminal, truncated, _ = env.step(a)
            
            current_length += 1
            if (length_cap is not None and current_length > length_cap) or \
                terminal or truncated:
                done = True     

            o = next_o   
        
        self._h = None
        if verbose: print("End Trajectory \n\n")
        return trajectory, hiddens
    
    def _get_action_with_input_mask_softmax(self, x_tensor, gru_state, mask_f=None, mask_g=None, mask_a=None):
        prob_actions, logits, next_gru_state = self._masked_input_forward_softmax(x_tensor, gru_state, mask_f, mask_g, mask_a)
        a = torch.argmax(prob_actions).item()
        return a, logits, next_gru_state
    
    def _get_action_with_input_mask_softmax_cahced(self, x_tensor, gru_state, mask_f=None, mask_g=None, mask_a=None):
        x_array = x_tensor.detach().cpu().numpy().tobytes()
        gru_array = gru_state.detach().cpu().numpy().tobytes()
        if (x_array, gru_array) in self.option_cache:
            prob_actions, logits, next_gru_state = self.option_cache[(x_array, gru_array)]
        else:
            prob_actions, logits, next_gru_state = self._masked_input_forward_softmax(x_tensor, gru_state, mask_f, mask_g, mask_a)
            self.option_cache[(x_array, gru_array)] = (prob_actions, logits, next_gru_state)
        a = torch.argmax(prob_actions).item()
        return a, logits, next_gru_state
    
    def run_with_input_mask_softmax(self, envs, mask_f=None, mask_g=None, mask_a=None, max_size_sequence=30):
        """
        Runs the model with provided masks on provided environments.
        Args:
            envs: environments to run the model on.
            mask_f (Tensor, optional): mask applied on feature extractor's input
            mask_g (Tensor, optional): mask applied on GRU's hidden states
            mask_a (Tensor, optional): mask applied on actor's NN input
            max_size_sequence (int, optional): maximum length for trajectory. Default: 30
        Returns:
            Trajecroty created by running model with masks on envs.
        """
        trajectory = Trajectory()

        length = 0
        gru_state = self.init_hidden().squeeze(0)
        if mask_g is not None:
            gru_state = self._masked_hidden_state_softmax(gru_state, mask_g)
        dones = torch.zeros(1)
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over()[0]:
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits, gru_state = self._get_action_with_input_mask_softmax(x_tensor, gru_state, mask_f, mask_g, mask_a)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over()[0]:
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits, gru_state = self._get_action_with_input_mask_softmax(x_tensor, gru_state, mask_f, mask_g, mask_a)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory
