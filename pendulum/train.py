#!/usr/bin/env python3

import gym
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch
from model import Actor, Critic
import sys

def t(s):
    return torch.from_numpy(s).float()

def clip_grads(module, max_grad_norm):
    torch.nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

class Trainer:
    def __init__(self):
        self.GAMMA = 0.99
        self.EPISODES_PER_EPOCH = 2000
        self.EPOCHS = 5
        
        self.env = gym.make("Pendulum-v0")

        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high)
        self.critic = Critic(self.env.observation_space.shape[0])
        try:
            self.actor.load_state_dict(torch.load('actor_pendulum.pth'))
            self.critic.load_state_dict(torch.load('critic_pendulum.pth'))
        except:
            print("Starting with new weights")

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.scheduler_actor = StepLR(self.optim_actor, step_size=1, gamma=0.5)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=5e-3)
        self.scheduler_critic = StepLR(self.optim_critic, step_size=1, gamma=0.5)
        self.max_grad_norm = 0.3

    def train(self):
        s, a = None, None
        MAX = self.env.observation_space.high
        if all(MAX == np.inf):
            MAX = 1.0

        for epoch in range(self.EPOCHS):
            for i_episode in range(self.EPISODES_PER_EPOCH):
                done = False
                total_rew = 0
                s = self.env.reset() / MAX
                while not done:
                    a = self.actor(t(s)).sample().detach()\
                            .clamp(self.env.action_space.low.min(), self.env.action_space.high.max())
                    new_s, rew, done, _= self.env.step(a.numpy())
                    new_s = new_s / MAX
                    delta = rew + (1-done) * self.GAMMA * self.critic(t(new_s)) - self.critic(t(s))
                    s = new_s 
                    total_rew += rew

                    critic_loss = delta.square()
                    self.optim_critic.zero_grad()
                    critic_loss.backward()
                    clip_grads(self.optim_critic, self.max_grad_norm)
                    self.optim_critic.step()

                    # actor
                    dist = self.actor(t(s))
                    log_prob = dist.log_prob(a)
                    actor_loss = (-log_prob * delta.detach()).mean()
                    self.optim_actor.zero_grad()
                    actor_loss.backward()
                    clip_grads(self.optim_actor, self.max_grad_norm)
                    self.optim_actor.step()
                    
                print("{} {}: {:0.2f}".format(epoch, i_episode, total_rew))
                if i_episode % 100 == 0:
                    torch.save(self.actor.state_dict(), 'actor_pendulum.pth'.format(self.env_name))
                    torch.save(self.critic.state_dict(), 'critic_pendulum.pth'.format(self.env_name))
            self.scheduler_actor.step()
            self.scheduler_critic.step()
        self.env.close()

def main():
    Trainer().train()

if __name__ == "__main__":
    main()

