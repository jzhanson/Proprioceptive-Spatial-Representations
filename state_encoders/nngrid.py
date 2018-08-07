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
    def __init__(self, args):
        super(NNGrid, self).__init__()
        self.grid_edge  = args['grid_edge']
        self.grid_scale = args['grid_scale']
        self.use_lidar  = args['grid_use_lidar']

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(10, self.grid_edge, self.grid_edge))

    def _coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    # TODO: pytorch'ify these functions so that input features are already Variables
    # this will allow future hybrid models to be fully-differentiable
    def forward(self, inputs):
        ob, info = inputs

        grid_state = torch.zeros(10, self.grid_edge, self.grid_edge)
        if ob.is_cuda:
            with torch.cuda.device(ob.get_device()):
                grid_state = grid_state.cuda()
        grid_state = Variable(grid_state)

        # Project raw state into grid, center grid at hull
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5

        # 1. For every body b in body config, get position (bx, by) and
        #   - Write angle of b to (0, bx, by)
        #   - Write angvel of b to (1,bx,by) in G
        #   - Write velx of b to (2,bx,by) in G
        #   - Write vely of b to (3,bx,by) in G
        #   - Write ground_contact of b to (4,bx,by) in G
        for b in info['bodies']:
            # b format:
            # pos_x, pos_y, ang,
            # ang_vel, lin_vel_x, lin_vel_y, contact
            pos_x, pos_y = b[0], b[1]
            f = Variable(torch.from_numpy(np.array(b[2:7])))

            # Round to nearest integer coordinates here
            grid_x, grid_y = self._coord_to_grid(pos_x, zero_x), self._coord_to_grid(pos_y, zero_y)
            # Not sure if these scalings apply for hull only or hull and legs
            grid_state[0:5, grid_x, grid_y] = f

        # 2. For every joint j in body configuration:
        #   - Get Position of Both Anchors of Joint (Ajx, Ajy), (Bjx, Bjy)
        #   - Write angle of j to (5,Ajx,Ajy),(7,Bjx,Bjy) in G
        #   - Write speed of j to (6,Ajx,Ajy),(8,Bjx,Bjy) in G
        for j in info['joints']:

            # j format:
            # A_x, A_y, B_x, B_y,
            # angle, speed
            A_pos_x, A_pos_y = j[0], j[1]
            B_pos_x, B_pos_y = j[2], j[3]
            f = Variable(torch.from_numpy(np.array(j[4:6])))

            # For each anchor position, write joint features
            A_grid_x, A_grid_y = self._coord_to_grid(A_pos_x, zero_x), self._coord_to_grid(A_pos_y, zero_y)
            B_grid_x, B_grid_y = self._coord_to_grid(B_pos_x, zero_x), self._coord_to_grid(B_pos_y, zero_y)

            grid_state[5:7, A_grid_x, A_grid_y] = f
            grid_state[7:9, B_grid_x, B_grid_y] = f

        # 3. Write lidar points
        #   - Write 1 at position of p2
        if self.use_lidar:
            for l in info['lidar']:
                p2_x, p2_y = self._coord_to_grid(l.p2[0], zero_x), self._coord_to_grid(l.p2[1], zero_y)
                
                grid_state[10,p2_x,p2_y] = 1.

        return grid_state[None]

    def initialize_memory(self):
        return None

    def reinitialize_memory(self, old_memory):
        return None
