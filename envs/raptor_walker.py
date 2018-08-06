import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import torch
from torch.autograd import Variable

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

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(box=(28/SCALE/2, 28/SCALE/2)),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0) # 0.99 bouncy

NECK1_FD = fixtureDef(
    shape=polygonShape(box=(20/SCALE/2, 20/SCALE/2)),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0) # 0.99 bouncy
NECK2_FD = fixtureDef(
    shape=polygonShape(box=(10/SCALE/2, 15/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)
NECK3_FD = fixtureDef(
    shape=polygonShape(box=(9/SCALE/2, 9/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

HEAD_FD = fixtureDef(
    shape=polygonShape(box=(25/SCALE/2, 12.5/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

TAIL1_FD = fixtureDef(
    shape=polygonShape(box=(15/SCALE/2, 20/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

TAIL2_FD = fixtureDef(
    shape=polygonShape(box=(22/SCALE/2, 10/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

TAIL3_FD = fixtureDef(
    shape=polygonShape(box=(22/SCALE/2, 8/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

TAIL4_FD = fixtureDef(
    shape=polygonShape(box=(22/SCALE/2, 5/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)

TAIL5_FD = fixtureDef(
    shape=polygonShape(box=(22/SCALE/2, 4/SCALE/2)),
    density=5.0,
    friction=0.,
    categoryBits=0x0020,
    maskBits=0x001, # collide only with ground
    restitution=0.0)


THIGH_FD = fixtureDef(
    shape=polygonShape(box=(10/SCALE/2, LEG_H/2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(7/SCALE/2, 24/SCALE/2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

FOOT_FD = fixtureDef(
    shape=polygonShape(box=(0.8*LEG_W/2, 20/SCALE/2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

TOES_FD = fixtureDef(
    shape=polygonShape(box=(0.7*LEG_W/2, 16/SCALE/2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)




class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class RaptorWalker(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=Box2D.b2Vec2(0,0))
        self.terrain = None
        self.hull = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        high = np.array([np.inf]*24)
        self.observation_space = spaces.Box(-high, high)

        self.reset()

        self.action_space = spaces.Box(
            np.array([-1]*len(self.joints)), np.array([+1]*len(self.joints)))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def reset(self, STATIC=False):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        body_color1 = (120/255.,140/255.,175/255.)
        body_color2 = (61/255., 72/255., 90/255.)

        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+2*LEG_H + 0.0
        self.hull = self.world.CreateStaticBody( #CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = HULL_FD,
        )
        self.hull.userData = self.hull
        self.hull.name = 'Hull'
        self.hull.depth = 0
        self.hull.ground_contact = False
        self.hull.color1 = body_color1
        self.hull.color2 = body_color2
        self.hull.can_touch_ground = False
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)
        self.hull.ApplyTorque(1000, True)
        self.hull.connected_body = [1, 3]
        self.hull.connected_joints = [0, 2]

        self.joints = []

        self.neck1 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=NECK1_FD,
        )
        self.neck1.userData = self.neck1
        self.neck1.name = 'Neck1'
        self.neck1.depth = 0
        self.neck1.can_touch_ground = False
        self.neck1.color1 = body_color1
        self.neck1.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.hull,
            bodyB=self.neck1,
            localAnchorA=(20/SCALE/2, -0/SCALE/2),
            localAnchorB=(-20/SCALE/2,  -10/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.25+0. if STATIC else -0.5,
            upperAngle = 0.25+0. if STATIC else +0.2,
        )
        rjd.depth = 0
        vert1 = self.world.CreateJoint(rjd)
        vert1.depth = 0
        self.joints.append(vert1)
        print(self.joints)
        print(rjd)
        jnt = self.joints[0]
        anchorA = jnt.anchorA
        anchorB = jnt.anchorB
        print(anchorA)
        print(anchorA - jnt.bodyA.position)
        print(anchorB - jnt.bodyB.position)

        self.neck2 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=NECK2_FD,
        )
        self.neck2.userData = self.neck2
        self.neck2.name = 'Neck2'
        self.neck2.depth = 0
        self.neck2.can_touch_ground = False
        self.neck2.color1 = body_color1
        self.neck2.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.neck1,
            bodyB=self.neck2,
            localAnchorA=(15/SCALE/2, -0/SCALE/2),
            localAnchorB=(-15/SCALE/2,  -6/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.1+0. if STATIC else -0.5,
            upperAngle = 0.1+0. if STATIC else +0.2,
        )
        rjd.depth = 0
        vert2 = self.world.CreateJoint(rjd)
        vert2.depth = 0
        self.joints.append(vert2)

        self.neck3 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=NECK3_FD,
        )
        self.neck3.userData = self.neck3
        self.neck3.name = 'Neck3'
        self.neck3.depth = 0
        self.neck3.can_touch_ground = False
        self.neck3.color1 = body_color1
        self.neck3.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.neck2,
            bodyB=self.neck3,
            localAnchorA=(10/SCALE/2, -0/SCALE/2),
            localAnchorB=(-10/SCALE/2,  -4/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.1+0. if STATIC else -0.5,
            upperAngle = 0.1+0. if STATIC else +0.2,
        )
        rjd.depth = 0
        vert3 = self.world.CreateJoint(rjd)
        vert3.depth = 0
        self.joints.append(vert3)

        self.neck4 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=NECK3_FD,
        )
        self.neck4.userData = self.neck4
        self.neck4.name = 'Neck4'
        self.neck4.depth = 0
        self.neck4.can_touch_ground = False
        self.neck4.color1 = body_color1
        self.neck4.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.neck3,
            bodyB=self.neck4,
            localAnchorA=(10/SCALE/2, -0/SCALE/2),
            localAnchorB=(-10/SCALE/2,  -0/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.1+0. if STATIC else -0.5,
            upperAngle = 0.1+0. if STATIC else +0.2,
        )
        rjd.depth = 0
        vert4 = self.world.CreateJoint(rjd)
        vert4.depth = 0
        self.joints.append(vert4)

        self.head = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=HEAD_FD,
        )
        self.head.userData = self.head
        self.head.name = 'Head'
        self.head.depth = 0
        self.head.can_touch_ground = False
        self.head.color1 = body_color1
        self.head.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.neck4,
            bodyB=self.head,
            localAnchorA=(10/SCALE/2, -0/SCALE/2),
            localAnchorB=(-10/SCALE/2,  -0/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = -1.1+0. if STATIC else -0.9,
            upperAngle = -1.1+0. if STATIC else +0.7,
        )
        rjd.depth = 0
        vert5 = self.world.CreateJoint(rjd)
        vert5.depth = 0
        self.joints.append(vert5)

        self.tail1 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=TAIL1_FD,
        )
        self.tail1.userData = self.tail1
        self.tail1.name = 'Tail1'
        self.tail1.depth = 0
        self.tail1.can_touch_ground = True
        self.tail1.color1 = body_color1
        self.tail1.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.hull,
            bodyB=self.tail1,
            localAnchorA=(-28/SCALE/2, -0/SCALE/2),
            localAnchorB=(15/SCALE/2,  -7/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.2+0. if STATIC else -0.5,
            upperAngle = 0.2+0. if STATIC else +0.5,
        )
        rjd.depth = 0
        vert6 = self.world.CreateJoint(rjd)
        vert6.depth = 0
        self.joints.append(vert6)

        self.tail2 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=TAIL2_FD,
        )
        self.tail2.userData = self.tail2
        self.tail2.name = 'Tail2'
        self.tail2.depth = 0
        self.tail2.can_touch_ground = True
        self.tail2.color1 = body_color1
        self.tail2.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.tail1,
            bodyB=self.tail2,
            localAnchorA=(-15/SCALE/2, -0/SCALE/2),
            localAnchorB=(15/SCALE/2,  -10/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.2+0. if STATIC else -0.5,
            upperAngle = 0.2+0. if STATIC else +0.5,
        )
        rjd.depth = 0
        vert7 = self.world.CreateJoint(rjd)
        vert7.depth = 0
        self.joints.append(vert7)

        self.tail3 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=TAIL3_FD,
        )
        self.tail3.userData = self.tail3
        self.tail3.name = 'Tail3'
        self.tail3.depth = 0
        self.tail3.can_touch_ground = True
        self.tail3.color1 = body_color1
        self.tail3.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.tail2,
            bodyB=self.tail3,
            localAnchorA=(-15/SCALE/2, -0/SCALE/2),
            localAnchorB=(15/SCALE/2,  -2/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.2+0. if STATIC else -0.5,
            upperAngle = 0.2+0. if STATIC else +0.5,
        )
        rjd.depth = 0
        vert8 = self.world.CreateJoint(rjd)
        vert8.depth = 0
        self.joints.append(vert8)

        self.tail4 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=TAIL4_FD,
        )
        self.tail4.userData = self.tail4
        self.tail4.name = 'Tail4'
        self.tail4.depth = 0
        self.tail4.can_touch_ground = True
        self.tail4.color1 = body_color1
        self.tail4.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.tail3,
            bodyB=self.tail4,
            localAnchorA=(-15/SCALE/2, -0/SCALE/2),
            localAnchorB=(15/SCALE/2,  -2/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = -0.15+0. if STATIC else -0.5,
            upperAngle = -0.15+0. if STATIC else +0.5,
        )
        rjd.depth = 0
        vert9 = self.world.CreateJoint(rjd)
        vert9.depth = 0
        self.joints.append(vert9)

        self.tail5 = self.world.CreateDynamicBody(
            position=(init_x+(28+18)/SCALE/2, init_y+8/SCALE/2),
            fixtures=TAIL5_FD,
        )
        self.tail5.userData = self.tail5
        self.tail5.name = 'Tail5'
        self.tail5.depth = 0
        self.tail5.can_touch_ground = True
        self.tail5.color1 = body_color1
        self.tail5.color2 = body_color2
        rjd = revoluteJointDef(
            bodyA=self.tail4,
            bodyB=self.tail5,
            localAnchorA=(-15/SCALE/2, -0/SCALE/2),
            localAnchorB=(15/SCALE/2,  -2/SCALE/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed = 1,
            lowerAngle = 0.2+0. if STATIC else -0.5,
            upperAngle = 0.2+0. if STATIC else +0.5,
        )
        rjd.depth = 0
        vert10 = self.world.CreateJoint(rjd)
        vert10.depth = 0
        self.joints.append(vert10)

        leg_color1 = [(90/255.,105/255.,120/255.),
                      (163/255.,177/255.,188/255.)]


        self.body = [self.neck1, self.neck2, self.neck3, self.neck4, self.head, self.tail1, self.tail2, self.tail3, self.tail4, self.tail5]

        self.legs = []
        for i in [-1,+1]:
            thigh = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = 0.1,
                fixtures = THIGH_FD
                )
            thigh.userData = thigh
            thigh.name = 'Thigh'+str(i)
            thigh.depth = (i+1)//2
            thigh.can_touch_ground = True
            thigh.color1 = leg_color1[(i+1)//2]
            thigh.color2 = body_color2
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=thigh,
                localAnchorA=(0, -0/SCALE/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = 0.2+0. if STATIC else -0.8,
                upperAngle = 0.2+0. if STATIC else +1.1,
                )
            thigh.ground_contact = False
            self.legs.append(thigh)
            rjd.depth = (i+1)//2
            hip = self.world.CreateJoint(rjd)
            hip.depth = (i+1)//2
            self.joints.append(hip)

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - 24/SCALE/2),
                angle = 0.1,
                fixtures = LOWER_FD
                )
            lower.userData = lower
            lower.name = 'Lower'+str(i)
            lower.depth = (i+1)//2
            lower.can_touch_ground = True
            lower.color1 = leg_color1[(i+1)//2]
            lower.color2 = body_color2
            rjd = revoluteJointDef(
                bodyA=thigh,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, 24/SCALE/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.5+0. if STATIC else -0.8,
                upperAngle = -1.5+0. if STATIC else +0.5,
                )
            lower.ground_contact = False
            rjd.depth =(i+1)//2
            knee = self.world.CreateJoint(rjd)
            knee.depth = (i+1)//2
            self.legs.append(lower)
            self.joints.append(knee)

            foot = self.world.CreateDynamicBody(
                position = (init_x, init_y - 24/SCALE/2),
                angle = (i*0.05),
                fixtures = FOOT_FD
                )
            foot.userData = foot
            foot.name = 'Foot'+str(i)
            foot.depth = (i+1)//2
            foot.can_touch_ground = True
            foot.color1 = leg_color1[(i+1)//2]
            foot.color2 = body_color2
            rjd = revoluteJointDef(
                bodyA=lower,
                bodyB=foot,
                localAnchorA=(0, -24/SCALE/2),
                localAnchorB=(0, 20/SCALE/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = 1.6+0. if STATIC else -1.5,
                upperAngle = 1.6+0. if STATIC else +0.9,
            )
            foot.ground_contact = False
            rjd.depth =(i+1)//2
            ankle = self.world.CreateJoint(rjd)
            ankle.depth = (i+1)//2
            self.legs.append(foot)
            self.joints.append(ankle)

            toes = self.world.CreateDynamicBody(
                position = (init_x, init_y - 24/SCALE/2),
                angle=(i*0.05),
                fixtures = TOES_FD
            )
            toes.userData = toes
            toes.name = 'Toes'+str(i)
            toes.depth = (i+1)//2
            toes.can_touch_ground = True
            toes.color1 = leg_color1[(i+1)//2]
            toes.color2 = body_color2
            rjd = revoluteJointDef(
                bodyA=foot,
                bodyB=toes,
                localAnchorA=(0, -24/SCALE/2),
                localAnchorB=(0,  16/SCALE/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = 1.2+0. if STATIC else -0.8,
                upperAngle = 1.2+0. if STATIC else +0.8,
            )
            toes.ground_contact = False
            rjd.depth =(i+1)//2
            ball = self.world.CreateJoint(rjd)
            ball.depth = (i+1)//2
            self.legs.append(toes)
            self.joints.append(ball)

        self.drawlist = [self.hull] + self.terrain + self.legs + self.body # + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        ob, _, _, info = self.step(np.zeros(len(self.joints)))
        return ob, info

    def _get_state(self):
        return [0] * 24
        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*self.hull.linearVelocity.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*self.hull.linearVelocity.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==24

        return state

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = True  # Should be easier as well
        if control_speed:
            for i, j in enumerate(self.joints):
                j.motorSpeed = float(SPEED_HIP * np.clip(action[i], -1, 1))
            #self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
            #self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            #self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
            #self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)
        #self.world.Step(0.00001, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        # Cache lidar results
        for  i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)


        state = self._get_state()

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True

        return np.array(state), reward, done, self._info_dict()

    def _info_dict(self):
        return {}
        # Construct the info dict so that the grid state space and action space can be built
        info = {}
        info['hull_x'] = self.hull.position.x
        info['hull_y'] = self.hull.position.y
        info['bodies'] = []
        body_parts = [self.hull] + self.legs
        for b_index in range(len(body_parts)):
            b = body_parts[b_index]
            info['bodies'].append((
                b.position.x, b.position.y, b.angle,
                2.0*b.angularVelocity/FPS,
                0.3*b.linearVelocity.x*(VIEWPORT_W/SCALE)/FPS,
                0.3*b.linearVelocity.y*(VIEWPORT_H/SCALE)/FPS,
                1.0 if b.ground_contact else 0,
                # Depth: 0.0 if "front", 1.0 if "back", hull is considered front and second leg is back leg
                0.0 if b_index < 3 else 1.0,
                b.connected_body,
                b.connected_joints
            ))
        info['joints'] = []
        for j_index in range(len(self.joints)):
            j = self.joints[j_index]
            even = j_index % 2 == 0
            info['joints'].append((
                j.anchorA.x, j.anchorA.y, j.anchorB.x, j.anchorB.y,
                j.angle + (0.0 if even else 1.0),
                j.speed / (SPEED_HIP if even else SPEED_KNEE),
                # Depth: second set of joints are in back
                0.0 if j_index < 2 else 1.0,
                j.connected_body,
                j.connected_joints
            ))
        return info

    def coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    def get_zeros(self):
        return self.hull.position.x - self.grid_scale / 2, self.hull.position.y - self.grid_scale / 2

    def _draw_stategrid(self, model):
        self.grid_edge = model.senc_nngrid.grid_edge
        self.grid_scale = model.senc_nngrid.grid_scale
        self.grid_square_edge = self.grid_scale / self.grid_edge

        # Draw state grid around hull and highlight squares with a nonzero channel
        fill_empty = True
        show_grid = True

        zero_x, zero_y = self.get_zeros()
        filled_in_squares = []

        body_color = (1, 1, 1, 0.5)
        joint_color = (1, 1, 1, 0.5)

        if fill_empty:
            vertices = [(zero_x, zero_y),
                        (zero_x + self.grid_scale, zero_y),
                        (zero_x + self.grid_scale, zero_y + self.grid_scale),
                        (zero_x, zero_y + self.grid_scale)]
            big_square = self.viewer.draw_polygon(vertices)
            big_square._color.vec4 = (0, 0, 0, 0.5)

        if show_grid:
            for i in range(self.grid_edge + 1):
                vertical = [(zero_x + self.grid_square_edge * i, zero_y), (zero_x + self.grid_square_edge * i, zero_y + self.grid_scale)]
                horizontal = [(zero_x, zero_y + self.grid_square_edge * i), (zero_x + self.grid_scale, zero_y + self.grid_square_edge * i)]
                self.viewer.draw_polyline(vertical, color=(0, 0, 0), linewidth=1)
                self.viewer.draw_polyline(horizontal, color=(0, 0, 0), linewidth=1)


        grid_channels = model.senc_nngrid((Variable(torch.from_numpy(np.array(self._get_state()))), self._info_dict()))
        grid_channels_sum = torch.sum(torch.squeeze(grid_channels), dim=0).data.numpy()

        for x in range(self.grid_edge):
            for y in range(self.grid_edge):
                if grid_channels_sum[x, y] != 0:
                    lower_left_x, lower_left_y = zero_x + self.grid_square_edge * x, zero_y + self.grid_square_edge * y
                    vertices = [(lower_left_x, lower_left_y),
                        (lower_left_x + self.grid_square_edge, lower_left_y),
                        (lower_left_x + self.grid_square_edge, lower_left_y + self.grid_square_edge),
                        (lower_left_x, lower_left_y + self.grid_square_edge)]
                    filled_in_square = self.viewer.draw_polygon(vertices)
                    # Make half transparent
                    filled_in_square._color.vec4 = body_color
                    filled_in_squares.append((lower_left_x, lower_left_y))

    def _draw_actiongrid(self, model, depth=-1):
        # Dimensions of action grid output by model not always the same as those of the state grid
        self.grid_edge = model.adec_nngrid.current_actiongrid.shape[2]
        self.grid_scale = model.adec_nngrid.grid_scale
        self.grid_square_edge = self.grid_scale / self.grid_edge

        # Draw action grid around hull and color in grayscale intensity of action
        show_grid = True

        zero_x, zero_y = self.get_zeros()

        if show_grid:
            for i in range(self.grid_edge + 1):
                vertical = [(zero_x + self.grid_square_edge * i, zero_y), (zero_x + self.grid_square_edge * i, zero_y + self.grid_scale)]
                horizontal = [(zero_x, zero_y + self.grid_square_edge * i), (zero_x + self.grid_scale, zero_y + self.grid_square_edge * i)]
                self.viewer.draw_polyline(vertical, color=(0, 0, 0), linewidth=1)
                self.viewer.draw_polyline(horizontal, color=(0, 0, 0), linewidth=1)

        # Depth of -1 means flatten depth layers and display
        if depth >= 0:
            current_actiongrid_layer = model.adec_nngrid.current_actiongrid[0, depth]
        else:
            current_actiongrid_layer = np.sum(model.adec_nngrid.current_actiongrid, axis=1)[0]
        max_sum = np.amax(current_actiongrid_layer)

        for x in range(self.grid_edge):
            for y in range(self.grid_edge):
                lower_left_x, lower_left_y = zero_x + self.grid_square_edge * x, zero_y + self.grid_square_edge * y
                vertices = [(lower_left_x, lower_left_y),
                    (lower_left_x + self.grid_square_edge, lower_left_y),
                    (lower_left_x + self.grid_square_edge, lower_left_y + self.grid_square_edge),
                    (lower_left_x, lower_left_y + self.grid_square_edge)]
                filled_in_square = self.viewer.draw_polygon(vertices)
                # Set intensity to the relative value of channels
                square_fill = current_actiongrid_layer[x, y] / max_sum
                filled_in_square._color.vec4 = (square_fill, square_fill, square_fill, 0.5)

    def render(self, mode='human', model=None, show_stategrid=False, show_actiongrid=False, actiongrid_depth=-1):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        if show_stategrid:
            self._draw_stategrid(model)
        if show_actiongrid:
            self._draw_actiongrid(model, depth=actiongrid_depth)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class RaptorWalkerHardcore(RaptorWalker):
    hardcore = True

if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = RaptorWalker()
    env.reset(STATIC=True)
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        env.render()
        a = np.zeros(env.action_space.shape)+1.
        s, r, done, info = env.step(a)
        if done:
            s, info = env.reset()
        continue
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
