from __future__ import division
import gym
import numpy as np
from collections import deque
from gym import spaces


class frame_stack:
    def __init__(self, enc, args):
        self.enc = enc
        self.stack_frames = args.stack_frames
        self.frames = deque([], maxlen=self.stack_frames)
        self.obs_norm = MaxMinFilter()
        #NormalizedEnv() alternative or can just not normalize observations as environment is already kinda normalized

    def reset(self, ob, info):
        ob = self.enc.reset(ob, info)
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self.observation()

    def encode(self, ob, info):
        ob = self.enc.encode(ob, info)
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        self.frames.append(ob)
        return self.observation(), rew, done, info

    def observation(self):
        assert len(self.frames) == self.stack_frames
        return np.stack(self.frames, axis=0)

class motion_blur(gym.Wrapper):
    def __init__(self, enc, args):
        self.enc = enc
        self.num_blur = args.blur_frames
        self.blur_discount = args.blur_discount
        self.blur_frames = deque([], maxlen=self.num_blur)

    def reset(self, ob, info):
        ob = self.enc.reset(ob, info)
        ob = np.float32(ob)
        # Need to normalize?
        for i in range(self.num_blur):
            self.blur_frames.appendleft(ob * (self.blur_discount ** i))
        return self.observation()

    def encode(self, ob, info):
        ob = self.enc.encode(ob, info)
        ob = np.float32(ob)
        self.blur_frames.append(ob)
        blurred = self.observation()
        # TODO(josh): recreating a deque every step is not very efficient, can consider using slicing + appending a np.array
        self.blur_frames = deque(map(lambda frame: frame * self.blur_discount, self.blur_frames), maxlen=self.num_blur)
        return blurred, rew, done, info

    def observation(self):
        assert len(self.blur_frames) == self.num_blur
        return np.sum(self.blur_frames, axis=0)


class MaxMinFilter():
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs


class NormalizedEnv():
    def __init__(self):
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def __call__(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
