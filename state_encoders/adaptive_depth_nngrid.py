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
        self.grid_cells_per_unit  = args['grid_cells_per_unit']

        # Keep track of min/max points we've seen
        self.min_x, self.max_x = 0, 0
        self.min_y, self.max_y = 0, 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 1, 1))

    def _coord_to_grid_x(self, x_coord, x_zero):
        return round((x_coord - x_zero) * self.grid_cells_per_unit)

    def _coord_to_grid_y(self, y_coord, y_zero):
        return round((y_coord - y_zero) * self.grid_cells_per_unit)

    def _update_bounds(self, info):
        unit_zero_x, unit_zero_y = info['hull_x'], info['hull_y']
        for b in info['bodies']:
            pos_x, pos_y = b[0] - unit_zero_x, b[1] - unit_zero_y
            if self.min_x > pos_x:
                self.min_x = pos_x
            if self.max_x < pos_x:
                self.max_x = pos_x
            if self.min_y > pos_y:
                self.min_y = pos_y
            if self.max_y < pos_y:
                self.max_y = pos_y

        for j in info['joints']:
            A_pos_x, A_pos_y = j[0] - unit_zero_x, j[1] - unit_zero_y
            if self.min_x > A_pos_x:
                self.min_x = A_pos_x
            if self.max_x < A_pos_x:
                self.max_x = A_pos_x
            if self.min_y > A_pos_y:
                self.min_y = A_pos_y
            if self.max_y < A_pos_y:
                self.max_y = A_pos_y

            B_pos_x, B_pos_y = j[2] - unit_zero_x, j[3] - unit_zero_y
            if self.min_x > B_pos_x:
                self.min_x = B_pos_x
            if self.max_x < B_pos_x:
                self.max_x = B_pos_x
            if self.min_y > B_pos_y:
                self.min_y = B_pos_y
            if self.max_y < B_pos_y:
                self.max_y = B_pos_y

    # TODO: pytorch'ify these functions so that input features are already Variables
    # this will allow future hybrid models to be fully-differentiable
    def forward(self, inputs):
        ob, info = inputs

        # Update min/max grid boundaries
        self._update_bounds(info)

        unit_zero_x, unit_zero_y = info['hull_x'], info['hull_y']

        # Create the gridstate based on the observed min/max values
        grid_unit_width  = 2 * max(self.max_x, abs(self.min_x))
        grid_unit_height = 2 * max(self.max_y, abs(self.min_y))
        print(grid_unit_width, grid_unit_height)
        grid_cell_width  = math.ceil(grid_unit_width  * self.grid_cells_per_unit)+1
        grid_cell_height = math.ceil(grid_unit_height * self.grid_cells_per_unit)+1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, grid_cell_width, grid_cell_height))

        grid_state = torch.zeros(self.observation_space.shape)
        if ob.is_cuda:
            with torch.cuda.device(ob.get_device()):
                grid_state = grid_state.cuda()
        grid_state = Variable(grid_state)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, grid_cell_width, grid_cell_height))

        # Calculate the grid anchor point
        # This is the center cell of the grid
        grid_anchor_x = round(0.5 * grid_cell_width)
        grid_anchor_y = round(0.5 * grid_cell_height)

        # Project raw state onto grid, center grid at hull
        zero_x, zero_y = info['hull_x'] - grid_unit_width * 0.5, info['hull_y'] - grid_unit_height * 0.5

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
            grid_x, grid_y = self._coord_to_grid_x(pos_x, zero_x), self._coord_to_grid_y(pos_y, zero_y)
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

            print(grid_state.size())
            print(A_pos_x, info['hull_x'], A_pos_x - info['hull_x'], self.max_x)

            # For each anchor position, write joint features
            A_grid_x, A_grid_y = self._coord_to_grid_x(A_pos_x, zero_x), self._coord_to_grid_y(A_pos_y, zero_y)
            B_grid_x, B_grid_y = self._coord_to_grid_x(B_pos_x, zero_x), self._coord_to_grid_y(B_pos_y, zero_y)

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

        return grid_state[None], (grid_anchor_x, grid_anchor_y)

    def initialize_memory(self):
        return None

    def reinitialize_memory(self, old_memory):
        return None
