#!/usr/bin/env python3

import gym
import numpy as np
import torch
from model import Actor, Critic

def t(s):
    return torch.from_numpy(s).float()

class Player:
    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        self.actor = Actor(3, 1, self.env.action_space.high[0])
        try:
            self.actor.load_state_dict(torch.load('actor_pendulum.pth'))
        except:
            raise Exception("No weights found")

    def play(self):
        MAX = self.env.observation_space.high
        done = False
        total_rew = 0
        s = self.env.reset() / MAX

        while not done:
            self.env.render()
            a = self.actor(t(s))
            s, rew, done, _= self.env.step(a.detach().numpy())
            s = s / MAX
            total_rew += rew

        self.env.close()
        print(total_rew)

def main():
    Player().play()

if __name__ == "__main__":
    main()


