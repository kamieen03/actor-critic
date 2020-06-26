from torch.nn import LeakyReLU, Linear, Tanh, Module, Sequential, Parameter
import torch

class Actor(Module):
    def __init__(self, state_dim, action_space_dim, a_max):
        super().__init__()
        self.mu_params = Sequential(
            Linear(state_dim, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, action_space_dim),
            Tanh()
        )
        self.logstd_params = Parameter(torch.ones(action_space_dim) * 0.2, requires_grad=True)
        self.a_max = a_max
    
    def forward(self, state):
        mu = self.mu_params(state) * self.a_max
        std = self.logstd_params.exp()
        std = std.clamp(0.0001, self.a_max)
        dist = torch.distributions.normal.Normal(mu, std)
        a = dist.rsample()
        a = a.clamp(-self.a_max, self.a_max)
        return a


class Critic(Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = Sequential(
            Linear(state_dim, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 1)
        )
    
    def forward(self, state):
        return self.model(state)
