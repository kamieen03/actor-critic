from torch.nn import LeakyReLU, Linear, Tanh, Module, Sequential, Parameter
import torch

class Actor(Module):
    def __init__(self, state_dim, action_space_dim, a_max):
        super().__init__()
        self.model = Sequential(
            Linear(state_dim, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, action_space_dim),
            Tanh()
        )
    
    def forward(self, state):
        return self.model(state) * 5


class Critic(Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = Sequential(
            Linear(state_dim + action_dim, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat((state, action), 1))
