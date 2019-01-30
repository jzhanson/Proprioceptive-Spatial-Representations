from __future__ import division
import gym
import math
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
        self.project_to_grid = args['project_to_grid']

    def _coords_to_grid(self, coord_x, coord_y, zero_x, zero_y):
        grid_space_x = (coord_x - zero_x) / self.grid_scale * self.grid_edge
        grid_space_y = (coord_y - zero_y) / self.grid_scale * self.grid_edge
        if self.project_to_grid:
            # Project into the grid by finding equation of line from zeroes to
            # the point and picking the intersection between the edges that lies
            # within bounds

            # y = self.grid_edge, 1 / m * y = x
            grid_top_x = grid_space_x / grid_space_y * self.grid_edge
            # x = self.grid_edge, y = m * x
            grid_right_y = grid_space_y / grid_space_x * self.grid_edge
            if grid_top_x < self.grid_edge:
                return (math.floor(grid_top_x), self.grid_edge - 1)
            elif grid_right_y < self.grid_edge:
                return (self.grid_edge - 1, math.floor(grid_right_y))
            else:
                return (self.grid_edge - 1, self.grid_edge - 1)
        else:
            return (min(math.floor(grid_space_x), self.grid_edge - 1),
                min(math.floor(grid_space_y), self.grid_edge - 1))

    def forward(self, inputs):
        action, info = inputs

        decoded_action = torch.zeros(action.size(0), self.action_space.shape[0])
        if action.is_cuda:
            with torch.cuda.device(action.get_device()):
                decoded_action = decoded_action.cuda()
        decoded_action = Variable(decoded_action)

        # Extract raw action from grid, centered at hull
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5
        for j_index, j in enumerate(info['joints']):
            # Alternatively, we can average the two anchor positions instead of just using anchorA
            A_pos_x, A_pos_y = j[0], j[1]
            grid_x, grid_y = self._coords_to_grid(A_pos_x, A_pos_y, zero_x, zero_y)
            decoded_action[0, j_index] = action[0, 0, grid_x, grid_y]
        return decoded_action
