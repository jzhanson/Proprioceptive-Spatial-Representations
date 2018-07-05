from __future__ import division
import gym
import numpy as np
from gym import spaces

class StateEncoder:
    def __init__(self, env, args):
        self.observation_space = env.observation_space

    def reset(self):
        pass

    def encode(self, ob, info):
        return ob
