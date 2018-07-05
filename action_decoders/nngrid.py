from __future__ import division
import gym
import numpy as np
from gym import spaces

class ActionDecoder:
    def __init__(self, args):
        self.action_space = env.action_space
        self.grid_edge  = args.grid_edge
        self.grid_scale = args.grid_scale

        self.action_space = spaces.Box(
            low=-1, high=-1, shape=(4, self.grid_edge, self.grid_edge))

    def _coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    def decode(self, action):
        # Extract raw action from grid, centered at hull
        decoded_action = np.zeros(self.action_space.shape)
        zero_x, zero_y = info['zero_x'], info['zero_y']

        for j in range(len(info['joints'])):
            # Alternatively, we can average the two anchor positions instead of just using anchorA
            A_pos_x, A_pos_y = j[0], j[1]
            grid_x, grid_y = self._coord_to_grid(A_pos_x, zero_x), self._coord_to_grid(A_pos_y, zero_y)
            decoded_action[j] = action[j, grid_x, grid_y]
        return decoded_action
