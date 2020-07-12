#!/usr/bin/env python3

import gym
import sys
import torch
from walker.model import Actor as WalkerActor
from pendulum.model import Actor as PendulumActor


def main():
    problem = sys.stdin.readline().strip()
    if problem == 'Pendulum-v0':
        _env = gym.make("Pendulum-v0")
        actor = PendulumActor(_env.observation_space.shape[0],
                _env.action_space.shape[0], _env.action_space.high)
        actor.load_state_dict(torch.load('pendulum/actor_pendulum.pth', map_location=torch.device('cpu')))
    else:
        _env = gym.make("BipedalWalker-v3")
        actor = WalkerActor(_env.observation_space.shape[0],
                _env.action_space.shape[0], _env.action_space.high)
        actor.load_state_dict(torch.load('walker/actor_walker.pth', map_location=torch.device('cpu')))
    
    N = int(sys.stdin.readline())
    with torch.no_grad():
        for _ in range(N):
            while True:
                inp = sys.stdin.readline().strip()
                if inp == 'END':
                    break
                s = inp[1:-1].split()
                s = [float(x) for x in s]
                s = torch.tensor(s)
                if problem == 'Pendulum-v0':
                    MAX = _env.observation_space.high
                    s = s/MAX
                    a = actor(s).sample().clamp(_env.action_space.low.min(), _env.action_space.high.max())
                else:
                    a = actor(s)
                str_a = str(a.tolist()).replace(',','')
                print(str_a, flush=True)

if __name__ == '__main__':
    main()
