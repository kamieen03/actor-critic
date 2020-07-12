#!/usr/bin/env python3

import gym
import numpy as np
import torch
from model import Actor, Critic
import sys

def t(s):
    return torch.from_numpy(s).float()

class Player:
    def __init__(self):
        self.env = gym.make("BipedalWalker-v3")
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high)
        try:
            self.actor.load_state_dict(torch.load('actor_walker.pth'))
        except:
            raise Exception("No weights found")

    def play(self):
        done = False
        total_rew = 0
        s = self.env.reset()

        while not done:
            self.env.render()
            a = self.actor(t(s)).detach()
            s, rew, done, _= self.env.step(a.numpy())
            s = s
            total_rew += rew
        self.env.close()
        print(total_rew)

def main():
    Player().play()

if __name__ == "__main__":
    main()


