from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from .convlstm_layer import ConvLSTM

from common.utils import norm_col_init, weights_init, weights_init_mlp

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

        self.conv1 = nn.Conv2d(self.n_frames*self.input_size[0], 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.convlstm = ConvLSTM(32, 32, 3, stride=1, padding=1)

        self.critic_linear = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.actor_linear  = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.actor_linear2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)

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
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        hx, cx = self.convlstm(x, (hx, cx))
        return hx, cx

    def forward(self, inputs):
        x, (hx, cx) = inputs

        batch_size = x.size(0)
        x = x.view(batch_size, 
                   self.n_frames*self.input_size[0], 
                   self.input_size[1], self.input_size[2])

        hx, cx = self._convlstmforward(x, hx, cx)
        x = hx
        
        critic_out = self.critic_linear(x).view(batch_size, self.output_size).mean(-1)
        actor_out = F.softsign(self.actor_linear(x)).view(batch_size, self.output_size)
        actor_out2 = self.actor_linear2(x).view(batch_size, self.output_size)

        return critic_out, actor_out, actor_out2, (hx, cx)

    def initialize_memory(self):
        if next(self.parameters()).is_cuda:
            return (Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2]).cuda()),
                    Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2]).cuda()))
        return (Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2])),
                Variable(torch.zeros(1, 32, self.input_size[1], self.input_size[2])))

    def reinitialize_memory(self, old_memory):
        old_hx, old_cx = old_memory
        return (Variable(old_hx.data),
                Variable(old_cx.data))
