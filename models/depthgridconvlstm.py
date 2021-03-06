from __future__ import division
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from state_encoders.wrappers import FrameStack
from state_encoders.depth_nngrid import NNGrid as senc_NNGrid
from action_decoders.depth_nngrid import NNGrid as adec_NNGrid

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

        # Action postprocessing
        self.adec_nngrid = adec_NNGrid(action_space, args)

        self.observation_space = observation_space
        self.action_space      = action_space

        self.input_size  = self.senc_nngrid.observation_space.shape
        self.output_size = int(np.prod(self.action_space.shape))

        _s = [32, 64, 128, 128]
        self.convlstm1 = ConvLSTM(self.frame_stack.n_frames*self.input_size[0], 32, 4, stride=1, padding=1)
        self.convlstm2 = ConvLSTM( 32,  64, 3, stride=1, padding=1)
        self.convlstm3 = ConvLSTM( 64, 128, 3, stride=1, padding=1)
        self.convlstm4 = ConvLSTM(128, 128, 3, stride=1, padding=1)
        self.convlstm = [
            self.convlstm1,
            self.convlstm2,
            self.convlstm3,
            self.convlstm4,
        ]
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
        self._convh0_module = nn.ParameterList(self.convh0)
        self._convc0_module = nn.ParameterList(self.convc0)
 
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
        #print(np.sum([torch.norm(ch0).item() for ch0 in self.convh0]),
        #      np.sum([torch.norm(cc0).item() for cc0 in self.convc0]))
        use_gpu = next(self.parameters()).is_cuda
        return (
            #self.convh0,
            #self.convc0,
            # <!> DO NOT REMOVE BELOW CODE <!>
            # Below code is needed to fix a strange bug in graph backprop
            # TODO(eparisot): debug this further (low priority, might be pytorch..)
            #[ch0 for ch0 in self.convh0],
            #[cc0 for cc0 in self.convc0],
            [Variable(torch.zeros(ch0.size()).cuda()) if use_gpu else Variable(torch.zeros(ch0.size())) for ch0 in self.convh0],
            [Variable(torch.zeros(cc0.size()).cuda()) if use_gpu else Variable(torch.zeros(cc0.size())) for cc0 in self.convc0],
            self.frame_stack.initialize_memory())

    def reinitialize_memory(self, old_memory):
        old_convhx, old_convcx, old_frames = old_memory
        return (
            [Variable(chx.data) for chx in old_convhx],
            [Variable(ccx.data) for ccx in old_convcx],
            self.frame_stack.reinitialize_memory(old_frames))
