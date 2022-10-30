import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Normalizer(nn.Module):
    # Normalize the states as seen by different agents
    # The normalized state can be input by networks
    # common to all agents
    def __init__(self, base_state_size, nstacks, num_agents, seed):
        super(Normalizer, self).__init__()
        self.seed = torch.manual_seed(seed)        
        self.fc = nn.Linear(num_agents, base_state_size, bias=False)
        self.nstacks = nstacks
        self.base_state_size = base_state_size
        self.num_agents = num_agents
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.weight.data.uniform_(*hidden_init(self.fc))

    def forward(self, states, agent_ids):
        agent_ids_oh = F.one_hot(agent_ids, self.num_agents).float()
        norm_weights = self.fc(agent_ids_oh).reshape((-1, self.base_state_size))
        #if agent_ids.size()[0] > 2:
        #    print(agent_ids_oh)
        #    print("-------")
        states = states.reshape((-1, self.nstacks, self.base_state_size)).permute(1,0,2)
        states = torch.mul(states, norm_weights).permute(1,0,2).reshape((-1,self.nstacks*self.base_state_size))
        return states

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, use_bn=False, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1: # Batchnorm1d requires N(=1)xC
            state = state.unsqueeze(0)
        
        x = self.fc1(state)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, use_bn=False, fcs1_units=400, fc2_units=300, fc3_units=150):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm1d(fcs1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        if self.use_bn:
            xs = self.bn1(xs)
        xs = F.relu(xs)
        
        x = torch.cat((xs, action), dim=1)

        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        if self.use_bn:
            x= self.bn3(x)
        x = F.relu(x)

        return self.fc4(x)
