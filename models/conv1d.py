from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from a3g.utils import norm_col_init, weights_init, weights_init_mlp

# Early-fusion Conv1D + LSTM
# All frames stacks, passed to 1D convnet then LSTM
class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space, n_frames):
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space      = action_space

        self.n_frames    = n_frames
        self.input_size  = int(np.prod(self.observation_space.shape))
        self.output_size = int(np.prod(self.action_space.shape))

        self.conv1 = nn.Conv1d(self.n_frames, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        dummy_input = Variable(torch.zeros(1, self.n_frames, self.input_size))
        dummy_conv_output = self._convforward(dummy_input)

        self.lstm = nn.LSTMCell(dummy_conv_output.nelement(), 128)
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, self.output_size)
        self.actor_linear2 = nn.Linear(128, self.output_size)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def _convforward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        return x

    def forward(self, inputs):
        x, (hx, cx) = inputs

        batch_size = x.size(0)
        x = x.view(batch_size, self.n_frames, self.input_size)

        x = self._convforward(x)
        x = x.view(x.size(0), -1)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)

    def initialize_memory(self):
        if next(self.parameters()).is_cuda:
            return (Variable(torch.zeros(1, 128).cuda()),
                    Variable(torch.zeros(1, 128).cuda()))
        return (Variable(torch.zeros(1, 128)),
                Variable(torch.zeros(1, 128)))

    def reinitialize_memory(self, old_memory):
        old_hx, old_cx = old_memory
        return (Variable(old_hx.data),
                Variable(old_cx.data))
