from __future__ import division
import gym
import numpy as np
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class NNGrid(torch.nn.Module):
    def __init__(self, action_space, args):
        super(NNGrid, self).__init__()

        self.action_space = action_space
        self.grid_edge  = args['grid_edge']
        self.grid_scale = args['grid_scale']

    def _coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    def forward(self, inputs):
        action, info = inputs

        # Expose current action grid to be passed to env for rendering
        self.current_actiongrid = action.data.numpy()

        decoded_action = torch.zeros(action.size(0), self.action_space.shape[0])
        if action.is_cuda:
            with torch.cuda.device(action.get_device()):
                decoded_action = decoded_action.cuda()
        decoded_action = Variable(decoded_action)

        # Extract raw action from grid, centered at hull
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5
        for j_index, j in enumerate(info['joints']):
            # j format:
            # A_x, A_y, B_x, B_y,
            # angle, speed, depth
            A_pos_x, A_pos_y = j[0], j[1]
            d = int(j[6])

            # Take action at grid position of AnchorA
            # Alternatively, we can average the two anchor positions instead of just using anchorA
            grid_x, grid_y = self._coord_to_grid(A_pos_x, zero_x), self._coord_to_grid(A_pos_y, zero_y)
            decoded_action[0, j_index] = action[0, d, grid_x, grid_y]
        return decoded_action
