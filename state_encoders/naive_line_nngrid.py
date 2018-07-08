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
        self.grid_edge  = args.grid_edge
        self.grid_scale = args.grid_scale

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, self.grid_edge, self.grid_edge))

    def _coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    def _dda_draw(self, grid_state, start, end, start_vals, end_vals, channels):
        if start == end:
            return
        start_x, start_y = start
        end_x, end_y = end
        start_channel, end_channel = channels
        differences = (end_vals - start_vals).astype(float)
        values = start_vals.astype(float)
        # To avoid division by 0
        if start_x == end_x:
            x = float(start_x)
            interval = abs(end_y - start_y)
            # Re-fills in endpoints
            for y in range(start_y, end_y+1):
                grid_state[start_channel:end_channel, round(x), y] = Variable(torch.from_numpy(values))
                values = values + differences / interval
            return
        m = (float(end_y) - float(start_y)) / (float(end_x) - float(start_x))
        if abs(m) > 1:
            x = float(start_x)
            interval = abs(end_y - start_y)
            # Re-fills in endpoints
            for y in range(start_y, end_y+1):
                grid_state[start_channel:end_channel, round(x), y] = Variable(torch.from_numpy(values))
                x = x + 1. / float(m)
                values = values + differences / interval
        else:
            y = float(start_y)
            interval = abs(end_x - start_x)
            for x in range(start_x, end_x+1):
                grid_state[start_channel:end_channel, x, round(y)] = Variable(torch.from_numpy(values))
                y = y + float(m)
                values = values + differences / interval

    # TODO: pytorch'ify these functions so that input features are already Variables
    # this will allow future hybrid models to be fully-differentiable
    def _draw_lines(self, grid_state, info):
        # First, connect body parts to each other. Don't route through joints yet
        zero_x, zero_y = info['hull_x'] - self.grid_scale * 0.5, info['hull_y'] - self.grid_scale * 0.5
        for b in info['bodies']:
            pos_x, pos_y = b[0], b[1]
            grid_x, grid_y = self._coord_to_grid(pos_x, zero_x), self._coord_to_grid(pos_y, zero_y)

            for b_neighbor_index in b[8]:
                b_neighbor = info['bodies'][b_neighbor_index]
                neighbor_pos_x, neighbor_pos_y = b_neighbor[0], b_neighbor[1]
                neighbor_grid_x, neighbor_grid_y = self._coord_to_grid(neighbor_pos_x, zero_x), self._coord_to_grid(neighbor_pos_y, zero_y)

                # Write to front/back channels
                start_ind = int(b[7]) * 5
                self._dda_draw(
                    grid_state,
                    (grid_x, grid_y),
                    (neighbor_grid_x, neighbor_grid_y),
                    np.array(b[2:7]),
                    np.array(b_neighbor[2:7]),
                    (start_ind, start_ind + 5)
                )

        for j in info['joints']:
            A_pos_x, A_pos_y = j[0], j[1]
            A_grid_x, A_grid_y = self._coord_to_grid(A_pos_x, zero_x), self._coord_to_grid(A_pos_y, zero_y)
            B_pos_x, B_pos_y = j[2], j[3]
            B_grid_x, B_grid_y = self._coord_to_grid(B_pos_x, zero_x), self._coord_to_grid(B_pos_y, zero_y)

            for j_neighbor_index in j[8]:
                j_neighbor = info['joints'][j_neighbor_index]
                A_neighbor_pos_x, A_neighbor_pos_y = j_neighbor[0], j_neighbor[1]
                A_neighbor_grid_x, A_neighbor_grid_y = self._coord_to_grid(A_neighbor_pos_x, zero_x), self._coord_to_grid(A_neighbor_pos_y, zero_y)
                B_neighbor_pos_x, B_neighbor_pos_y = j_neighbor[2], j_neighbor[3]
                B_neighbor_grid_x, B_neighbor_grid_y = self._coord_to_grid(B_neighbor_pos_x, zero_x), self._coord_to_grid(B_neighbor_pos_y, zero_y)

                start_ind = 10 + int(j[6]) * 4
                self._dda_draw(
                    grid_state,
                    (A_grid_x, A_grid_y),
                    (A_neighbor_grid_x, A_neighbor_grid_y),
                    np.array(j[4:6]),
                    np.array(j_neighbor[4:6]),
                    (start_ind, start_ind + 2)
                )
                self._dda_draw(
                    grid_state,
                    (B_grid_x, B_grid_y),
                    (B_neighbor_grid_x, B_neighbor_grid_y),
                    np.array(j[4:6]),
                    np.array(j_neighbor[4:6]),
                    (start_ind, start_ind + 2)
                )
        return grid_state


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
            # ang_vel, lin_vel_x, lin_vel_y, contact, depth,
            # connected_body, connected_joints
            pos_x, pos_y = b[0], b[1]
            f = Variable(torch.from_numpy(np.array(b[2:7])))
            d = b[7]
            d_var = Variable(torch.tensor(b[7]))

            # Round to nearest integer coordinates here
            grid_x, grid_y = self._coord_to_grid(pos_x, zero_x), self._coord_to_grid(pos_y, zero_y)
            if d == 0:
                grid_state[0:5, grid_x, grid_y] = f
                grid_state[18, grid_x, grid_y] = d_var
            else:
                grid_state[5:10, grid_x, grid_y] = f
                grid_state[19, grid_x, grid_y] = Variable(torch.tensor(1.0))


        # 2. For every joint j in body configuration:
        #   - Get Position of Both Anchors of Joint (Ajx, Ajy), (Bjx, Bjy)
        #   - Write angle of j to (10,Ajx,Ajy),(12,Bjx,Bjy) in G if front and (14,Ajx,Ajy),(16,Bjx,Bjy) if back
        #   - Write speed of j to (11,Ajx,Ajy),(13,Bjx,Bjy) in G if front and (15,Ajx,Ajy),(17,Bjx,Bjy) if back
        #   - Write depth of j to (18,Ajx,Ajy),(18,Bjx,Bjy) in G if front and (19,Ajx,Ajy),(19,Bjx,Bjy) if back
        for j in info['joints']:
            # j format:
            # A_x, A_y, B_x, B_y,
            # angle, speed, depth,
            # connected_body, connected_joints
            A_pos_x, A_pos_y = j[0], j[1]
            B_pos_x, B_pos_y = j[2], j[3]
            f = Variable(torch.from_numpy(np.array(j[4:6])))
            d = j[6]
            d_var = Variable(torch.tensor(j[6]))

            # For each anchor position, write joint features
            A_grid_x, A_grid_y = self._coord_to_grid(A_pos_x, zero_x), self._coord_to_grid(A_pos_y, zero_y)
            B_grid_x, B_grid_y = self._coord_to_grid(B_pos_x, zero_x), self._coord_to_grid(B_pos_y, zero_y)

            if d == 0:
                grid_state[10:12, A_grid_x, A_grid_y] = f
                grid_state[12:14, B_grid_x, B_grid_y] = f
                grid_state[18, A_grid_x, A_grid_y] = d_var
                grid_state[18, B_grid_x, B_grid_y] = d_var
            else:
                grid_state[14:16, A_grid_x, A_grid_y] = f
                grid_state[16:18, B_grid_x, B_grid_y] = f
                grid_state[19, A_grid_x, A_grid_y] = Variable(torch.tensor(1.0))
                grid_state[19, B_grid_x, B_grid_y] = Variable(torch.tensor(1.0))

        return self._draw_lines(grid_state, info)[None]

    def initialize_memory(self):
        return None

    def reinitialize_memory(self, old_memory):
        return None
