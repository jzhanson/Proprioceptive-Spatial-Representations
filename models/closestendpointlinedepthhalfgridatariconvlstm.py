from __future__ import division
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from state_encoders.wrappers import FrameStack
from state_encoders.closestendpoint_line_nngrid import NNGrid as senc_NNGrid

from .convlstm_layer import ConvLSTM

from common.utils import norm_col_init, weights_init, weights_init_mlp

# Early-fusion Conv1D + LSTM
# All frames stacks, passed to 1D convnet then LSTM
class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space, n_frames, args):
        super(ActorCritic, self).__init__()

        # State preprocessing
        self.senc_nngrid = senc_NNGrid(args)
        self.frame_stack = FrameStack(n_frames)

        self.observation_space = observation_space
        self.action_space      = action_space

        self.input_size  = self.senc_nngrid.observation_space.shape
        self.output_size = int(np.prod(self.action_space.shape))

        _s = [32, 64, 128]
        self.convlstm1 = ConvLSTM(self.frame_stack.n_frames*self.input_size[0], _s[0], 4, stride=2, padding=0)
        self.convlstm2 = ConvLSTM(_s[0], _s[1],  3, stride=2, padding=0)
        self.convlstm3 = ConvLSTM(_s[1], _s[2], 3, stride=1, padding=0)

        self.convlstm = [
            self.convlstm1,
            self.convlstm2,
            self.convlstm3,
        ]

        _is = (n_frames*self.input_size[0],)+self.input_size[1:]
        self.memsizes = []
        for i in range(3):
            _is = self.convlstm[i]._spatial_size_output_given_input((1,)+_is)
            _is = (_s[i],)+_is
            self.memsizes.append(copy.deepcopy(_is))

        self.lstm = nn.LSTMCell(np.prod(self.memsizes[-1]), 128)

        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear  = nn.Linear(128, self.action_space.shape[0])
        self.actor_linear2 = nn.Linear(128, self.action_space.shape[0])

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def _convlstmforward(self, x, convhx, convcx):
        last_convhx = x
        for i in range(len(self.convlstm)):
            convhx[i], convcx[i] = self.convlstm[i](last_convhx, (convhx[i], convcx[i]))
            last_convhx = convhx[i]
        return convhx, convcx

    def forward(self, inputs):
        ob, info, (convhx, convcx, hx, cx, frames) = inputs

        # Get the grid state from vectorized input
        x = self.senc_nngrid((ob, info))

        # Stack it
        x, frames = self.frame_stack((x, frames))

        # Resize to correct dims for convnet
        batch_size = x.size(0)
        x = x.view(batch_size,
                   self.frame_stack.n_frames*self.input_size[0],
                   self.input_size[1], self.input_size[2])
        convhx, convcx = self._convlstmforward(x, convhx, convcx)
        x = convhx[-1]

        x = x.view(1, -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        critic_out = self.critic_linear(x)
        actor_out = F.softsign(self.actor_linear(x))
        actor_out2 = self.actor_linear2(x)

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (convhx, convcx, hx, cx, frames)

    def initialize_memory(self):
        if next(self.parameters()).is_cuda:
            return (
                [Variable(torch.zeros((1,)+self.memsizes[i]).cuda()) for i in range(3)],
                [Variable(torch.zeros((1,)+self.memsizes[i]).cuda()) for i in range(3)],
                Variable(torch.zeros(1, 128).cuda()),
                Variable(torch.zeros(1, 128).cuda()),
                self.frame_stack.initialize_memory())
        return (
            [Variable(torch.zeros((1,)+self.memsizes[i])) for i in range(3)],
            [Variable(torch.zeros((1,)+self.memsizes[i])) for i in range(3)],
            Variable(torch.zeros(1, 128)),
            Variable(torch.zeros(1, 128)),
            self.frame_stack.initialize_memory())

    def reinitialize_memory(self, old_memory):
        old_convhx, old_convcx, old_hx, old_cx, old_frames = old_memory
        return (
            [Variable(chx.data) for chx in old_convhx],
            [Variable(ccx.data) for ccx in old_convcx],
            Variable(old_hx.data),
            Variable(old_cx.data),
            self.frame_stack.reinitialize_memory(old_frames))
