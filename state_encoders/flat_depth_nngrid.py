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

class FlatDepthNNGrid(torch.nn.Module):
    def __init__(self, args):
        super(FlatDepthNNGrid, self).__init__()
        self.grid_edge  = args['grid_edge']
        self.grid_scale = args['grid_scale']
        self.use_lidar  = args['grid_use_lidar']
        self.project_to_grid = args['project_to_grid']

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(args['observation_dim'] + 2 * self.grid_edge
                * self.grid_edge,))

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

    # TODO: pytorch'ify these functions so that input features are already Variables
    # this will allow future hybrid models to be fully-differentiable
    def forward(self, inputs):
        ob, info = inputs

        grid_state = torch.zeros(2, self.grid_edge, self.grid_edge)
        if ob.is_cuda:
            with torch.cuda.device(ob.get_device()):
                grid_state = grid_state.cuda()
        grid_state = Variable(grid_state)

        # Project raw state into grid, center grid at hull
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5

        # 1. For every body b in body config, get position (bx, by) and
        #   - Write 1 to (bx, by)
        for b in info['bodies']:
            # b format:
            # pos_x, pos_y, ang,
            # ang_vel, lin_vel_x, lin_vel_y, contact
            pos_x, pos_y = b[0], b[1]
            d = b[7]

            # Round to nearest integer coordinates here
            grid_x, grid_y = self._coords_to_grid(pos_x, pos_y, zero_x, zero_y)
            # Not sure if these scalings apply for hull only or hull and legs
            grid_state[d, grid_x, grid_y] = 1.


        # 2. For every joint j in body configuration:
        #   - Get Position of Both Anchors of Joint (Ajx, Ajy), (Bjx, Bjy)
        #   - Write 1 to (Ajx, Ajy) and (Bjx, Bjy)
        for j in info['joints']:
            # j format:
            # A_x, A_y, B_x, B_y,
            # angle, speed
            A_pos_x, A_pos_y = j[0], j[1]
            B_pos_x, B_pos_y = j[2], j[3]
            d = j[6]

            A_grid_x, A_grid_y = self._coords_to_grid(A_pos_x, A_pos_y, zero_x, zero_y)
            B_grid_x, B_grid_y = self._coords_to_grid(B_pos_x, B_pos_y, zero_x, zero_y)
            grid_state[d, A_grid_x, A_grid_y] = 1.
            grid_state[d, B_grid_x, B_grid_y] = 1.

        # 3. Write lidar points
        #   - Write 1 at position of p2
        if self.use_lidar:
            for l in info['lidar']:
                p2_x, p2_y = self._coords_to_grid(l.p2[0], l.p2[1], zero_x, zero_y)
                # By default, write to depth 0
                grid_state[0,p2_x,p2_y] = 1.

        return torch.cat((ob, grid_state.view(1, -1)), dim=1)

    def initialize_memory(self):
        return None

    def reinitialize_memory(self, old_memory):
        return None
