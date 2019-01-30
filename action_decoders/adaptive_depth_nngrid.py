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
        self.grid_cells_per_unit = args['grid_cells_per_unit']
        self.project_to_grid = args['project_to_grid']

        # Keep track of min/max points we've seen
        self.min_x, self.max_x = 0, 0
        self.min_y, self.max_y = 0, 0

        self.action_space = action_space

    # TODO(josh): integrate_coords_to_grid and get rid of grid_cells_per_unit
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

    def _coord_to_grid(self, coord, zero):
        return round((coord - zero) * self.grid_cells_per_unit)

    def _update_bounds(self, info):
        # TODO(josh): make the bounds scale down, to some preset minimum
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

    def forward(self, inputs):
        action, info = inputs

        # Update min/max grid boundaries
        self._update_bounds(info)

        # Expose current action grid to be passed to env for rendering
        self.current_actiongrid = action.data.cpu().numpy()

        decoded_action = torch.zeros(action.size(0), self.action_space.shape[0])
        if action.is_cuda:
            with torch.cuda.device(action.get_device()):
                decoded_action = decoded_action.cuda()
        decoded_action = Variable(decoded_action)

        # Extract raw action from grid, centered at hull
        grid_unit_width  = self.max_x - self.min_x
        grid_unit_height = self.max_y - self.min_y
        zero_x, zero_y = info['hull_x'] - grid_unit_width * 0.5, info['hull_y'] - grid_unit_height * 0.5
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
