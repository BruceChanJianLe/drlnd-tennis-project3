#!/usr/bin/env python
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py

import numpy as np
import copy
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck noise process."""

    def __init__(self, size, use_cuda, mu=0., theta=1, sigma=.15, decay_rate=.95):
        """Initialize parameters and noise process."""
        self.mu = np.array([mu for i in range(size)])
        self.theta = theta
        self.sigma = sigma
        self.decay = 1
        self.decay_rate = decay_rate
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, as_tensor=True):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.random(len(x))
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        if as_tensor:
            return torch.tensor(self.state * self.decay).float().to(device)
        else:
            return self.state * self.decay
        
    def decay_step(self):
        """Decrease the amplitude of the noise"""
        self.decay *= self.decay_rate