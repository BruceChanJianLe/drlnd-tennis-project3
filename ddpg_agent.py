#!/usr/bin/env python
import numpy as np
import random

from ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ounoise import OUNoise

# Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 300        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size=8, action_size=2, num_agents=2, noise_theta=0, noise_sigma=0, noise_decay_rate=1, random_seed=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): The number of agents sharing the common replay buffer
            noise_theta (float): The parameter theta in the Ornstein–Uhlenbeck process
            noise_sigma (float): The parameter sigma in the Ornstein–Uhlenbeck process
            noise_decay_rate (float): The decay rate in the Ornstein–Uhlenbeck process
            cuda (bool): If True, try to use the GPU
        """
        self.state_size = state_size
        self.action_size = action_size
        # self.seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.ounoise = OUNoise(action_size, True, 0., noise_theta, noise_sigma, noise_decay_rate)
        
    def decide(self, states, use_target=False, as_tensor=False, add_noise=True, autograd=False):
        """Returns actions for given states as per current policy.
        
        Parameters
        ==========
        states (np.Ndarray or torch.Tensor): The states that the actor will evaluate
        use_target (bool): Use the target actor network if True, else use the local actor network
        as_tensor (bool): Return actions as a tensor if True, else return as numpy array
        add_noise (bool): Add noise from the OU process to the actor's output
        autograd (bool): Activate autograd when evaluating the states
        """       
        # Check input type
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(device)
        
        # Select appropiate network
        if use_target:
            network = self.actor_target
        else:
            network = self.actor_local   
        
        # To autograd or not to autograd, that is the question
        if autograd:
            actions = network(states)
        else:
            network.eval()
            with torch.no_grad():
                actions = network(states)
            network.train()
        
        # Noise
        if add_noise:
            actions = actions + self.ounoise.sample()
                
        # Clipping & casting
        if as_tensor:
            actions = torch.clamp(actions, -1, 1)
        else:
            actions = np.clip(actions.cpu().data.numpy(), -1, 1)

        return actions

        
    def learn(self, experiences, next_actions, current_actions, agent_number):
        """Update actor and critics using the sampled experiences and the updated actions.
        Params
        ======
            experiences (Tuple of (states, actions, rewards, next_states, dones)):
                The experiences sampled from the replay buffer. Each tuple element
                is a list of tensors, where the ith tensor corresponds to the ith 
                agent.
            next_actions (list of tensors):
                The target actors' output, for all next_states in experiences.
                The ith tensor corresponds to the output from the ith agent.
            current_actions (list of tensors):
                The local actors' output, for all states in experiences.
                The ith tensor corresponds to the output from the ith agent.
            agent_number (int):
                The index of the current agent, to extract the correct tensors
                from experiences.
        """
        # Extract and pre-process data
        states, actions, rewards, next_states, dones = experiences
        states = torch.cat(states, dim=1)
        actions = torch.cat(actions, dim=1)
        next_states = torch.cat(next_states, dim=1)
        rewards = rewards[agent_number]
        dones = dones[agent_number]
        next_actions = torch.cat(next_actions, dim=1)
        current_actions = torch.cat([ca if i == agent_number else ca.detach() 
                                     for i, ca in enumerate(current_actions)], dim=1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Clip gradient
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(states, current_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)    
    

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
