from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .convlstm_layer import ConvLSTM

from a3g.utils import norm_col_init, weights_init, weights_init_mlp

# Early-fusion Conv1D + LSTM
# All frames stacks, passed to 1D convnet then LSTM
class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space, n_frames):
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space      = action_space

        self.n_frames    = n_frames
        self.input_size  = self.observation_space.shape
        self.output_size = int(np.prod(self.action_space.shape))

        self.convlstm1 = ConvLSTM(self.n_frames*self.input_size[0], 32, 3, stride=1, padding=1)
        self.convlstm2 = ConvLSTM(32, 32, 3, stride=1, padding=1)
        self.convlstm3 = ConvLSTM(32, 32, 3, stride=1, padding=1)
        self.convlstm4 = ConvLSTM(32, 32, 3, stride=1, padding=1)

        self.critic_linear = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.actor_linear  = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.actor_linear2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        # TODO(eparisot): does this make sense for conv?
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

    def _convlstmforward(self, x, hx, cx):
        hx[0], cx[0] = self.convlstm1(x,     (hx[0],cx[0]))
        hx[1], cx[1] = self.convlstm2(hx[0], (hx[1],cx[1]))
        hx[2], cx[2] = self.convlstm3(hx[1], (hx[2],cx[2]))
        hx[3], cx[3] = self.convlstm4(hx[2], (hx[3],cx[3]))
        return hx, cx

    def forward(self, inputs):
        x, (hx, cx) = inputs

        batch_size = x.size(0)
        x = x.view(batch_size, 
                   self.n_frames*self.input_size[0], 
                   self.input_size[1], self.input_size[2])

        hx, cx = self._convlstmforward(x, hx, cx)
        x = hx[3]
        
        critic_out = self.critic_linear(x).view(batch_size, self.output_size).mean(-1)
        actor_out = F.softsign(self.actor_linear(x)).view(batch_size, self.output_size)
        actor_out2 = self.actor_linear2(x).view(batch_size, self.output_size)

        return critic_out, actor_out, actor_out2, (hx, cx)

    def initialize_memory(self):
        if next(self.parameters()).is_cuda:
            return ([Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2]).cuda()) for i in range(4)],
                    [Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2]).cuda()) for i in range(4)])
        return ([Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2])) for i in range(4)],
                [Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2])) for i in range(4)])

    def reinitialize_memory(self, old_memory):
        old_hx, old_cx = old_memory
        return ([Variable(old_hx_l.data) for old_hx_l in old_hx],
                [Variable(old_cx_l.data) for old_cx_l in old_cx])
