#!/usr/bin/env python3

import gym
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch
from model import Actor, Critic
from memory import ReplayMemory
import sys

def t(s):
    return torch.from_numpy(s).float()

def clip_grads(module, max_grad_norm):
    torch.nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def std_from_ep(ep):
    return 0.1
    return max(1-ep/10000, 0.01) * np.cos(ep*2*np.pi/200)**2 + 0.01

def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * tau + param.data * (1-tau))


class Trainer:
    def __init__(self):
        self.env = gym.make("BipedalWalker-v3")
        self.GAMMA = 0.99
        self.EPISODES = 1000
        self.UPDATES_PER_EPISODE = 100
        self.max_grad_norm = 0.3
        self.memory = ReplayMemory()
        self.tau = 0.995

        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high)
        self.actor_target = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high)
        self.critic = Critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.critic_target = Critic(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        try:
            self.actor.load_state_dict(torch.load('actor_walker.pth'))
            self.critic.load_state_dict(torch.load('critic_walker.pth'))
        except:
            print("Starting with new weights")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = self.actor.cuda()
        self.actor_target = self.actor_target.cuda()
        self.critic = self.critic.cuda()
        self.critic_target = self.critic_target.cuda()

        self.optim_actor = torch.optim.SGD(self.actor.parameters(), lr=1e-3, momentum=0.9)
        self.optim_critic = torch.optim.SGD(self.critic.parameters(), lr=1e-3, momentum=0.9)

    def train(self):
        for ep in range(self.EPISODES):
            self.play_episode(ep)
            self.update_params()
            if ep % 100 == 0:
                torch.save(self.actor.state_dict(), 'actor_walker.pth')
                torch.save(self.critic.state_dict(), 'critic_walker.pth')



    def play_episode(self, ep):
        total = 0
        s = t(self.env.reset()).cuda()
        rand = torch.distributions.Normal(0, std_from_ep(ep))
        while True:
            a = self.actor(s).detach() + rand.sample([4]).cuda()
            new_s, rew, done, _= self.env.step(a.cpu().numpy())
            new_s = t(new_s)
            done = 1 * done
            self.memory.push(s, a, rew, new_s, done)
            s = new_s.cuda()
            total += rew
            if done:
                break
        print("{}: {}".format(ep, total))

    def update_params(self):
        for _ in range(self.UPDATES_PER_EPISODE):
            batch = self.memory.sample()
            if batch is None:
                return
            s, a, r, new_s, done = batch

            with torch.no_grad():
                targets = r + self.GAMMA * (1-done) * self.critic_target(new_s, self.actor_target(new_s))
                targets = targets.float()
            self.optim_critic.zero_grad()
            qsa = self.critic(s, a)
            critic_loss = F.mse_loss(qsa, targets)
            critic_loss.backward()
            clip_grads(self.optim_critic, self.max_grad_norm)
            self.optim_critic.step()

            self.optim_actor.zero_grad()
            actor_loss = -self.critic(s, self.actor(s)).mean()
            actor_loss.backward()
            clip_grads(self.optim_actor, self.max_grad_norm)
            self.optim_actor.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)



def main():
    Trainer().train()

if __name__ == "__main__":
    main()

