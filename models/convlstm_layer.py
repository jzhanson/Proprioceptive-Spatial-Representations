from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from a3g.utils import weights_init

# ConvLSTM layer
class ConvLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, 4*out_channels, kernel_size,
                              stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias)

        sigmoid_gain = nn.init.calculate_gain('sigmoid')
        tanh_gain    = nn.init.calculate_gain('tanh')
        self.conv.weight[0*self.out_channels:1*self.out_channels].data.mul_(sigmoid_gain)
        self.conv.weight[1*self.out_channels:2*self.out_channels].data.mul_(sigmoid_gain)
        self.conv.weight[2*self.out_channels:3*self.out_channels].data.mul_(tanh_gain)
        self.conv.weight[3*self.out_channels:4*self.out_channels].data.mul_(sigmoid_gain)

        self.conv.bias.data.fill_(0)


    def forward(self, x, memory):
        hx, cx = memory

        _g = self.conv(x)
        gi = F.sigmoid(_g[:,0*self.out_channels:1*self.out_channels])
        gf = F.sigmoid(_g[:,1*self.out_channels:2*self.out_channels])
        gg = F.tanh   (_g[:,2*self.out_channels:3*self.out_channels])
        go = F.sigmoid(_g[:,3*self.out_channels:4*self.out_channels])

        cx = gf * cx + gi * gg
        hx = go * F.tanh(cx)

        return hx, cx
