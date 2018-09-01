import sys, math, random, json, copy, time
import numpy as np
import os
from os import listdir
from os.path import isfile, join

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import torch
from torch.autograd import Variable

from json_walker import JSONWalker

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

#LIDAR_RANGE = 160/SCALE
LIDAR_RANGE   = 160/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        for k in self.env.bodies:
            if self.env.bodies[k] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.bodies[k].ground_contact = True
                if not self.env.bodies[k].can_touch_ground:
                    self.env.game_over = True
    def EndContact(self, contact):
        for k in self.env.bodies:
            if self.env.bodies[k] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.bodies[k].ground_contact = False

# TODO(josh): make this class inherit from JSONWalker but override reset/init?
class RandomJSONWalker(JSONWalker):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        # Load the json randomly
        files_list = [f for f in listdir('box2d-json-gen') if isfile(join('box2d-json-gen', f))]
        chosen_json = os.path.join('box2d-json-gen', random.choice(files_list))
        print(chosen_json)
        super(RandomJSONWalker, self).__init__(chosen_json)

        self.reset()

    def reset(self):
        # Load the json randomly
        files_list = [f for f in listdir('box2d-json-gen') if isfile(join('box2d-json-gen', f))]
        chosen_json = random.choice(files_list)
        self.load_json('box2d-json-gen/' + chosen_json)

        return super(RandomJSONWalker, self).reset()
        

class RandomJSONWalkerHardcore(RandomJSONWalker):
    hardcore = True

if __name__=="__main__":

    # TODO(josh): add arguments for how many bodies to choose from, which types of bodies, etc
    body_number = random.randint(0, 9)

    env = RandomJSONWalker()

    steps = 0
    total_reward = 0
    a = np.array([0.0]*env.action_space.shape[0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        env.render()
        a = env.action_space.sample()
        #a = np.zeros(env.action_space.shape)
        time.sleep(0.2)
        _, r, done, info = env.step(a)
        if done:
            env.reset()
        continue
        # Build the state
        s = [
            env.bodies['Hull'].angle,
            2.0*env.bodies['Hull'].angularVelocity/FPS,
            0.3*env.bodies['Hull'].linearVelocity.x*(VIEWPORT_W/SCALE)/FPS,
            0.3*env.bodies['Hull'].linearVelocity.y*(VIEWPORT_H/SCALE)/FPS,
            env.joints['HullLeg-1Joint'].angle,
            env.joints['HullLeg-1Joint'].speed / env.joint_defs['HullLeg-1Joint']['Speed'],
            env.joints['Leg-1Lower-1Joint'].angle + 1.0,
            env.joints['Leg-1Lower-1Joint'].speed / env.joint_defs['Leg-1Lower-1Joint']['Speed'],
            1.0 if env.bodies['Lower-1'].ground_contact else 0.0,
            env.joints['HullLeg1Joint'].angle,
            env.joints['HullLeg1Joint'].speed / env.joint_defs['HullLeg1Joint']['Speed'],
            env.joints['Leg1Lower1Joint'].angle + 1.0,
            env.joints['Leg1Lower1Joint'].speed / env.joint_defs['Leg1Lower1Joint']['Speed'],
            1.0 if env.bodies['Lower1'].ground_contact else 0.0
        ]
        s += [l.fraction for l in env.lidar]

        if done:
            env.reset()
        #continue
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
