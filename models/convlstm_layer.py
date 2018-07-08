from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from common.utils import weights_init

# ConvLSTM layer
class ConvLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 self_kernel_size=3):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        self.conv_x = nn.Conv2d(in_channels, 4*out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.conv_h = nn.Conv2d(out_channels, 4*out_channels, self_kernel_size,
                                stride=1, padding=(self_kernel_size-1)//2,
                                bias=False)

        sigmoid_gain = nn.init.calculate_gain('sigmoid')
        tanh_gain    = nn.init.calculate_gain('tanh')
        self.conv_x.weight[0*self.out_channels:1*self.out_channels].data.mul_(sigmoid_gain)
        self.conv_x.weight[1*self.out_channels:2*self.out_channels].data.mul_(sigmoid_gain)
        self.conv_x.weight[2*self.out_channels:3*self.out_channels].data.mul_(tanh_gain)
        self.conv_x.weight[3*self.out_channels:4*self.out_channels].data.mul_(sigmoid_gain)
        self.conv_h.weight[0*self.out_channels:1*self.out_channels].data.mul_(sigmoid_gain)
        self.conv_h.weight[1*self.out_channels:2*self.out_channels].data.mul_(sigmoid_gain)
        self.conv_h.weight[2*self.out_channels:3*self.out_channels].data.mul_(tanh_gain)
        self.conv_h.weight[3*self.out_channels:4*self.out_channels].data.mul_(sigmoid_gain)

        if bias:
            self.conv_x.bias.data.fill_(0)


    def forward(self, x, memory):
        hx, cx = memory

        _g = self.conv_x(x) + self.conv_h(hx)
        gi = F.sigmoid(_g[:,0*self.out_channels:1*self.out_channels])
        gf = F.sigmoid(_g[:,1*self.out_channels:2*self.out_channels])
        gg = F.tanh   (_g[:,2*self.out_channels:3*self.out_channels])
        go = F.sigmoid(_g[:,3*self.out_channels:4*self.out_channels])

        cx = gf * cx + gi * gg
        hx = go * F.tanh(cx)

        return hx, cx

    def _spatial_size_output_given_input(self, input_size):
        x = torch.zeros(input_size)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        x = Variable(x)
        out = self.conv_x(x)
        x = None
        return tuple(out.size())[2:]
