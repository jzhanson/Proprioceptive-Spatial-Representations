from __future__ import division
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from state_encoders.wrappers import AdaptiveFrameStack as FrameStack
from state_encoders.adaptive_depth_nngrid import NNGrid as senc_NNGrid
from action_decoders.adaptive_depth_nngrid import NNGrid as adec_NNGrid

from .convlstm_layer import ConvLSTM

from common.utils import norm_col_init, weights_init, weights_init_mlp, recenter_old_grid

# Early-fusion Conv1D + LSTM
# All frames stacks, passed to 1D convnet then LSTM
class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space, n_frames, args):
        super(ActorCritic, self).__init__()

        # State preprocessing
        self.senc_nngrid = senc_NNGrid(args)
        self.frame_stack = FrameStack(n_frames)
        self.old_anchor = (self.senc_nngrid.grid_anchor_x, self.senc_nngrid.grid_anchor_y)

        # Action postprocessing
        self.adec_nngrid = adec_NNGrid(action_space, args)

        self.observation_space = observation_space
        self.action_space      = action_space

        self.input_size  = self.senc_nngrid.observation_space.shape
        self.output_size = int(np.prod(self.action_space.shape))

        _s = [32, 64, 128, 128]
        self.convlstm1 = ConvLSTM(self.frame_stack.n_frames*self.input_size[0], 32, 3, stride=1, padding=1)
        self.convlstm2 = ConvLSTM( 32,  64, 3, stride=1, padding=1)
        self.convlstm3 = ConvLSTM( 64, 128, 3, stride=1, padding=1)
        self.convlstm4 = ConvLSTM(128, 128, 3, stride=1, padding=1)
        self.convlstm = [
            self.convlstm1,
            self.convlstm2,
            self.convlstm3,
            self.convlstm4,
        ]

        # TODO(eparisot): add code that initializes new gridcells with learnable parameter vector
        _is = (n_frames*self.input_size[0],)+self.input_size[1:]
        self.convh0 = []
        self.convc0 = []
        self.memsizes = []
        for i in range(len(self.convlstm)):
            _is = self.convlstm[i]._spatial_size_output_given_input((1,)+_is)
            _is = (_s[i],)+_is
            self.memsizes.append(copy.deepcopy(_is))
            self.convh0.append(nn.Parameter(torch.zeros((1,)+self.memsizes[i])))
            self.convc0.append(nn.Parameter(torch.zeros((1,)+self.memsizes[i])))
            setattr(self, '_convh0_l'+str(i), self.convh0[i])
            setattr(self, '_convc0_l'+str(i), self.convc0[i])

        self.critic_linear = nn.Conv2d(128, 2, 3, stride=1, padding=1)
        self.actor_linear  = nn.Conv2d(128, 2, 3, stride=1, padding=1)
        self.actor_linear2 = nn.Conv2d(128, 2, 3, stride=1, padding=1)

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
        ob, info, (convhx, convcx, frames) = inputs

        # Get the grid state from vectorized input
        x, anchor = self.senc_nngrid((ob, info))
        change_in_size = np.any(self.input_size != self.senc_nngrid.observation_space.shape)
        self.input_size = self.senc_nngrid.observation_space.shape
        if self.old_anchor is None:
            self.old_anchor = (anchor[0], anchor[1])

        # Stack it
        x, frames = self.frame_stack((x, frames, anchor))

        # Resize to correct dims for convnet
        batch_size = x.size(0)
        x = x.view(batch_size,
                   self.frame_stack.n_frames*self.input_size[0],
                   self.input_size[1], self.input_size[2])

        # Was there a change in size or anchor?
        if change_in_size or (anchor[0] != self.old_anchor[0]) or (anchor[1] != self.old_anchor[1]):
            new_size = x.size()
            for i in range(len(convhx)):
                new_convhx = torch.zeros(convhx[i].size()[:2]+new_size[2:])
                new_convcx = torch.zeros(convcx[i].size()[:2]+new_size[2:])
                if convhx[i].is_cuda:
                    new_convhx = new_convhx.cuda()
                    new_convcx = new_convcx.cuda()
                new_convh0 = Variable(new_convhx.clone())
                new_convc0 = Variable(new_convcx.clone())
                new_convhx = Variable(new_convhx)
                new_convcx = Variable(new_convcx)
                convhx[i] = recenter_old_grid(convhx[i], self.old_anchor, new_convhx, anchor)
                convcx[i] = recenter_old_grid(convcx[i], self.old_anchor, new_convcx, anchor)
                if convhx[i] is not self.convh0[i]:
                    self.convh0[i] = recenter_old_grid(self.convh0[i], self.old_anchor, new_convh0, anchor)
                    self.convc0[i] = recenter_old_grid(self.convc0[i], self.old_anchor, new_convc0, anchor)
        self.old_anchor = (anchor[0], anchor[1])

        convhx, convcx = self._convlstmforward(x, convhx, convcx)
        x = convhx[-1]

        # Compute action mean, action var and value grid
        critic_out = self.critic_linear(x)
        actor_out = F.softsign(self.actor_linear(x))
        actor_out2 = self.actor_linear2(x)

        # Extract motor-specific values from action grid
        critic_out = self.adec_nngrid((critic_out, info)).mean(-1, keepdim=True)
        actor_out  = self.adec_nngrid((actor_out, info))
        actor_out2 = self.adec_nngrid((actor_out2, info))
        return critic_out, actor_out, actor_out2, (convhx, convcx, frames)

    def initialize_memory(self):
        return (
            self.convh0, self.convc0,
            self.frame_stack.initialize_memory())

    def reinitialize_memory(self, old_memory):
        old_convhx, old_convcx, old_frames = old_memory
        return (
            [Variable(chx.data) for chx in old_convhx],
            [Variable(ccx.data) for ccx in old_convcx],
            self.frame_stack.reinitialize_memory(old_frames))
