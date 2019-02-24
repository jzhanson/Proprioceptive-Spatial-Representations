import sys, math, random, json, copy, time
import numpy as np
from os import listdir
from os.path import isfile, join

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import torch
from torch.autograd import Variable

from envs.json_walker import JSONWalker

from generate_scripts.randomize_bodies import RandomizeBodies

class TrueRandomJSONWalker(JSONWalker):
    hardcore = False

    def __init__(self, params_jsonfile, truncate_state=False,
            max_state_dim=None, max_action_dim=None):
        with open(params_jsonfile) as f:
            self.args = json.load(f)

        # Make sure num_bodies is 1
        self.args['num_bodies'] = 1
        self.rb = RandomizeBodies(self.args['body_type'], self.args)

        # First json generated is a placeholder
        self.body = self._generate_json()
        super(TrueRandomJSONWalker, self).__init__(
            jsondata=self.body,
            truncate_state=truncate_state,
            max_state_dim=max_state_dim, max_action_dim=max_action_dim)

        self.reset()

    def _generate_json(self):
        body = self.rb.build_bodies(write_to_file=False)
        return body


    def reset(self):
        # Make new json body every episode
        self.load_dict(self._generate_json())
        return super(TrueRandomJSONWalker, self).reset()


class TrueRandomJSONWalkerHardcore(TrueRandomJSONWalker):
    hardcore = True

if __name__=="__main__":
    env = TrueRandomJSONWalker('datasets/params-bipedal-25p-1-12-segments-offcen.json')
    env.reset()

    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()

