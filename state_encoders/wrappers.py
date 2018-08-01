from __future__ import division
import gym
import copy
import numpy as np
from collections import deque
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from common.utils import recenter_old_grid

class MaxMinFilter(torch.nn.Module):
    def __init__(self):
        super(MaxMinFilter, self).__init__()
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def forward(self, x):
        obs = x.clamp(min=self.mn_d, max=self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs


class NormalizedEnv(torch.nn.Module):
    def __init__(self):
        super(NormalizedEnv, self).__init__()
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def forward(self, x):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            x.mean().item() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            x.std().item() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (x - unbiased_mean) / (unbiased_std + 1e-8)

class FrameStack(torch.nn.Module):
    def __init__(self, n_frames):
        super(FrameStack, self).__init__()
        self.n_frames = n_frames
        self.obs_norm = MaxMinFilter()
        #NormalizedEnv() alternative or can just not normalize observations as environment is already kinda normalized

    def forward(self, inputs):
        x, frames = inputs
        #x = self.obs_norm(x)
        frames.append(x[:,None])
        while len(frames) != self.n_frames:
            frames.append(x[:,None])
        x = torch.cat(list(frames), 1)
        return x, frames

    def initialize_memory(self):
        return deque([], maxlen=self.n_frames)

    def reinitialize_memory(self, old_memory):
        return deque(
            [Variable(e.data) for e in old_memory],
            maxlen=self.n_frames
        )

class AdaptiveFrameStack(torch.nn.Module):
    def __init__(self, n_frames):
        super(AdaptiveFrameStack, self).__init__()
        self.n_frames = n_frames
        self.obs_norm = MaxMinFilter()
        self.old_anchor = False
        #NormalizedEnv() alternative or can just not normalize observations as environment is already kinda normalized

    def forward(self, inputs):
        x, frames, new_anchor = inputs

        x = x[:,None]
        if len(frames) > 0 and (frames[-1].size(-1) != x.size(-1) or frames[-1].size(-2) != x.size(-2)):
            for i in range(len(frames)):
                new_frame = torch.zeros(x.size())
                if x.is_cuda:
                    new_frame = new_frame.cuda()
                new_frame = Variable(new_frame)
                frames[i] = recenter_old_grid(frames[i], self.old_anchor, new_frame, new_anchor)

        self.old_anchor = copy.deepcopy(new_anchor)

        frames.append(x)
        while len(frames) != self.n_frames:
            frames.append(x)
        x = torch.cat(list(frames), 1)
        return x, frames

    def initialize_memory(self):
        return deque([], maxlen=self.n_frames)

    def reinitialize_memory(self, old_memory):
        return deque(
            [Variable(e.data) for e in old_memory],
            maxlen=self.n_frames
        )


class MotionBlur(torch.nn.Module):
    def __init__(self, blur_frames):
        super(MotionBlur, self).__init__()
        self.blur_frames = blur_frames

    def forward(self, inputs):
        x, frames = inputs
        frames.append(x[:,None])
        while len(frames) != self.n_frames:
            frames.append(x[:,None])
        x = sum(list(frames))

        return x, frames

    def initialize_memory(self):
        return deque([], maxlen=self.blur_frames)

    def reinitialize_memory(self, old_memory):
        return deque(
            [Variable(e.data) for e in old_memory],
            maxlen=self.blur_frames
        )
