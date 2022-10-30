import numpy as np
import random
import copy
from collections import namedtuple, deque

from importlib import reload
import model
reload(model)
from model import Normalizer, Actor, Critic


import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 128        # minibatch size
LR_NORM = 1e-3          # learning rate of the normalizer
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-6     # L2 weight decay
ACT_FC1 = 256           # Hidden size of the actor network (layer 1)
ACT_FC2 = 128           # Hidden size of the actor network (layer 2)
CRI_FC1 = 256           # Hidden size of the critic network (layer 1)
CRI_FC2 = 128           # Hidden size of the critic network (layer 2)
CRI_FC3 = 128           # Hidden size of the critic network (layer 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class MADDPG():
    def __init__(self, num_agents, state_size, num_stacks, action_size,
                 random_seed, use_bn=False,
                 # Replay buffer size
                 buffer_size_init=int(1e5), buffer_size_final=int(1e6),
                 # Discount factor
                 gamma = 0.99,
                 # Noise scaling factor (epsilon): initial, min value and decay factor per episode
                 epsilon_init = 1.0, epsilon_min = 1e-2, epsilon_decay = 0.99,
                 # Target weights soft update factor (tau): initial, min value and decay factor per episode                 
                 tau_init = 1e-2, tau_min = 1e-3, tau_decay=0.95):
        """
        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of each state
            num_stacks (int): number of stacked base states (state_size is a multiple of num_stacks)
            action_size (int): dimension of each action
            random_seed (int)
            use_bn (bool): use batch norm or not
        """
        self.num_agents = num_agents
        self.state_size = state_size
        base_state_size = state_size // num_stacks
        self.action_size = action_size

        # Normalizer network (w/ Target network)
        self.normalizer_local = Normalizer(base_state_size, num_stacks, num_agents, random_seed)
        self.normalizer_target = Normalizer(base_state_size, num_stacks, num_agents, random_seed)
        self.normalizer_optimizer = optim.Adam(self.normalizer_local.parameters(), lr=LR_NORM)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, use_bn, ACT_FC1, ACT_FC2).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, use_bn, ACT_FC1, ACT_FC2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, use_bn, CRI_FC1, CRI_FC2, CRI_FC3).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, use_bn, CRI_FC1, CRI_FC2, CRI_FC3).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size_init, buffer_size_final, BATCH_SIZE, random_seed)

        # Noise process
        self.noise = []
        for a in range(num_agents):
            self.noise.append(OUNoise(action_size, random_seed+a))

        # Discount factor
        self.gamma = gamma

        # (decaying) epsilon modulating additional noise
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Weights soft udpate factor:
        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_decay = tau_decay

    def normalizer_loss(self):
        loss = 0
        for name, p in self.normalizer_local.named_parameters():
            if 'bias' not in name:
                loss += torch.pow(1 - torch.abs(p),2)
        return torch.mean(loss)

    def act(self, states, add_noise=True):
        actions=np.zeros((self.num_agents, self.action_size))
        states = torch.from_numpy(states).float().to(device)
        agent_ids = torch.arange(2, dtype=torch.long).reshape((2,1)).to(device)
        with torch.no_grad():
            states_norm = self.normalizer_local(states, agent_ids)
        for a in range(self.num_agents):
            with torch.no_grad():
                action = self.actor_local(states_norm[a,:]).cpu().data.numpy()
            if add_noise:
                action += self.epsilon * self.noise[a].sample()
            actions[a,:] = np.clip(action, -1, 1)
        return actions

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_tau(self):
        self.tau = max(self.tau_min, self.tau * self.tau_decay)

    def switch_replay_buffer_size(self):
        self.memory.switch_capacity()

    def reset(self):
        for a in range(self.num_agents):
            self.noise[a].reset()

    def step(self, states, actions, rewards, next_states, dones, n_train = 1):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if (len(self.memory) > BATCH_SIZE) and (n_train > 0):
            self.critic_local.train()
            self.actor_local.train()
            for t in range(n_train):
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        agent_ids, states, actions, rewards, next_states, dones = experiences

        # Normalize states
        next_states_norm_tgt = self.normalizer_target(next_states, agent_ids)
        states_norm_loc = self.normalizer_local(states, agent_ids)

        # ------- Normalizer only loss --------
        normalizer_loss = self.normalizer_loss()
        self.normalizer_optimizer.zero_grad()        
        normalizer_loss.backward(retain_graph=True)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states_norm_tgt)
        Q_targets_next = self.critic_target(next_states_norm_tgt, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states_norm_loc, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss (Maximize value predicted by critic)
        actions_pred = self.actor_local(states_norm_loc)
        actor_loss = -self.critic_local(states_norm_loc, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.normalizer_optimizer.step()

        critic_loss = None
        Q_targets = None

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.normalizer_local, self.normalizer_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size_init, buffer_size_final, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size_init)  # internal memory (deque)
        self.buffer_size_final = buffer_size_final
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["agent_id", "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def switch_capacity(self):
        if self.memory.maxlen != self.buffer_size_final:
            print("Switching replay buffer size from {} to {}".format(self.memory.maxlen,self.buffer_size_final))
            self.memory = deque(self.memory, maxlen=self.buffer_size_final)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        for a in range(states.shape[0]):
            e = self.experience(a, states[a,:], actions[a,:], rewards[a], next_states[a,:], dones[a])
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        agent_ids = torch.from_numpy(np.vstack([e.agent_id for e in experiences if e is not None])).long().to(device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (agent_ids, states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)