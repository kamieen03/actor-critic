#!/usr/bin/env python3

import gym
import numpy as np
import torch
from model import Actor, Critic
import sys

NV = False
N = 1

def t(s):
    return torch.from_numpy(s).float()

class Player:
    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space.high)
        try:
            self.actor.load_state_dict(torch.load('actor_pendulum.pth'))
        except:
            raise Exception("No weights found")

    def play(self):
        MAX = self.env.observation_space.high
        if all(MAX == np.inf):
            MAX = 1.0
        total_rew = 0
        for _ in range(N):
            done = False
            s = self.env.reset() / MAX
            while not done:
                if not NV:
                    self.env.render()
                a = self.actor(t(s)).sample().detach()\
                                    .clamp(self.env.action_space.low.min(), self.env.action_space.high.max())
                s, rew, done, _= self.env.step(a.numpy())
                s = s / MAX
                total_rew += rew

            self.env.close()
        print(total_rew/N)

def main():
    global NV, N
    if len(sys.argv) > 1 and sys.argv[1] == '-nv':
        NV = True
        if len(sys.argv) > 2:
            N = int(sys.argv[2])
    Player().play()

if __name__ == "__main__":
    main()


