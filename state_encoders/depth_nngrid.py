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
    def __init__(self, args):
        super(NNGrid, self).__init__()
        self.grid_edge  = args['grid_edge']
        self.grid_scale = args['grid_scale']
        self.use_lidar  = args['grid_use_lidar']
        self.project_to_grid = args['project_to_grid']

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(21, self.grid_edge, self.grid_edge))

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

        grid_state = torch.zeros(self.observation_space.shape)
        if ob.is_cuda:
            with torch.cuda.device(ob.get_device()):
                grid_state = grid_state.cuda()
        grid_state = Variable(grid_state)

        # Project raw state into grid, center grid at hull
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5

        # 1. For every body b in body config, get position (bx, by) and
        #   - Write angle of b to (0, bx, by) if front and (5, bx, by) if back
        #   - Write angvel of b to (1,bx,by) in G if front and (6, bx, by) if back
        #   - Write velx of b to (2,bx,by) in G if front and (7, bx, by) if back
        #   - Write vely of b to (3,bx,by) in G if front and (8, bx, by) if back
        #   - Write ground_contact of b to (4,bx,by) in G if front and (9, bx, by) if back
        #   - Write 1.0 to (18, bx, by) in G if front and (19, bx, by) if back
        for b in info['bodies']:
            # b format:
            # pos_x, pos_y, ang,
            # ang_vel, lin_vel_x, lin_vel_y, contact, depth
            pos_x, pos_y = b[0], b[1]
            f = Variable(torch.from_numpy(np.array(b[2:7])))
            d = b[7]

            # Round to nearest integer coordinates here
            grid_x, grid_y = self._coords_to_grid(pos_x, pos_y, zero_x, zero_y)
            if d == 0:
                grid_state[0:5, grid_x, grid_y] = f
                grid_state[18, grid_x, grid_y] = 1
            else:
                grid_state[5:10, grid_x, grid_y] = f
                grid_state[19, grid_x, grid_y] = 1


        # 2. For every joint j in body configuration:
        #   - Get Position of Both Anchors of Joint (Ajx, Ajy), (Bjx, Bjy)
        #   - Write angle of j to (10,Ajx,Ajy),(12,Bjx,Bjy) in G if front and (14,Ajx,Ajy),(16,Bjx,Bjy) if back
        #   - Write speed of j to (11,Ajx,Ajy),(13,Bjx,Bjy) in G if front and (15,Ajx,Ajy),(17,Bjx,Bjy) if back
        #   - Write depth of j to (18,Ajx,Ajy),(18,Bjx,Bjy) in G if front and (19,Ajx,Ajy),(19,Bjx,Bjy) if back
        for j in info['joints']:
            # j format:
            # A_x, A_y, B_x, B_y,
            # angle, speed, depth
            A_pos_x, A_pos_y = j[0], j[1]
            B_pos_x, B_pos_y = j[2], j[3]
            f = Variable(torch.from_numpy(np.array(j[4:6])))
            d = j[6]

            # For each anchor position, write joint features
            A_grid_x, A_grid_y = self._coords_to_grid(A_pos_x, A_pos_y, zero_x, zero_y)
            B_grid_x, B_grid_y = self._coords_to_grid(B_pos_x, B_pos_y, zero_x, zero_y)

            if d == 0:
                grid_state[10:12, A_grid_x, A_grid_y] = f
                grid_state[12:14, B_grid_x, B_grid_y] = f
                grid_state[18, A_grid_x, A_grid_y] = 1
                grid_state[18, B_grid_x, B_grid_y] = 1
            else:
                grid_state[14:16, A_grid_x, A_grid_y] = f
                grid_state[16:18, B_grid_x, B_grid_y] = f
                grid_state[19, A_grid_x, A_grid_y] = 1
                grid_state[19, B_grid_x, B_grid_y] = 1

        # 3. Write lidar points
        #   - Write 1 at position of p2
        if self.use_lidar:
            for l in info['lidar']:
                p2_x, p2_y = self._coords_to_grid(l.p2[0], l.p2[1], zero_x, zero_y)

                if (p2_x >= 0) and (p2_x < grid_state.size(1)) and (p2_y >= 0) and (p2_y < grid_size.size(2)):
                    grid_state[20,p2_x,p2_y] = 1.

        return grid_state[None]

    def initialize_memory(self):
        return None

    def reinitialize_memory(self, old_memory):
        return None
