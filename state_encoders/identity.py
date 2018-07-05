from __future__ import division
import gym
import numpy as np
from gym import spaces

class StateEncoder(gym.Wrapper):
    def __init__(self, env, args):
        super(StateEncoder, self).__init__(env)

    def reset(self):
        return env.reset()

    def step(self, action):
        return env.step(action)
