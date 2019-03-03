import sys, math, json, copy, time
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, weldJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import torch
from torch.autograd import Variable

# For rainbow actiongrid
import colorsys

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

class JSONWalker(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self, jsonfile=None, jsondata=None, truncate_state=False,
            max_state_dim=None, max_action_dim=None):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None

        self.prev_shaping = None

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.truncate_state = truncate_state

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

        if jsonfile is not None:
            with open(jsonfile) as f:
                self.load_dict(json.load(f))
        elif jsondata is not None:
            self.load_dict(jsondata)
        self.reset()

    def load_dict(self, jsondata):
        self.jsondata = jsondata
        self.fixture_defs = {}
        self.body_defs    = {}
        self.joint_defs   = {}
        self.linkage_defs = {}

        # Split the data so that we can create it later in order
        for k in self.jsondata.keys():
            if self.jsondata[k]['DataType'] == 'Fixture':
                self.fixture_defs[k] = copy.deepcopy(self.jsondata[k])
            elif self.jsondata[k]['DataType'] == 'DynamicBody':
                self.body_defs[k] = copy.deepcopy(self.jsondata[k])
            elif self.jsondata[k]['DataType'] == 'JointMotor':
                self.joint_defs[k] = copy.deepcopy(self.jsondata[k])
            elif self.jsondata[k]['DataType'] == 'Linkage':
                self.linkage_defs[k] = copy.deepcopy(self.jsondata[k])
            else:
                assert(False)

        self.enabled_joints_keys = [k for k in self.joint_defs.keys() if self.joint_defs[k]['EnableMotor']]
        if self.truncate_state:
            self.joints_to_report_keys = [k for k in self.joint_defs.keys() if self.joint_defs[k]['ReportState']]
            self.bodies_to_report_keys = [k for k in self.body_defs.keys() if self.body_defs[k]['ReportState']]
        else:
            self.joints_to_report_keys = [k for k in self.joint_defs.keys()]
            self.bodies_to_report_keys = [k for k in self.body_defs.keys()]
        num_enabled_joints = len(self.enabled_joints_keys)

        if self.max_state_dim is not None:
            high = np.array( [np.inf]*self.max_state_dim )
        else:
            high = np.array( [np.inf]*(5*len(self.bodies_to_report_keys)+2*len(self.joints_to_report_keys)+10) )

        if self.max_action_dim is not None:
            self.action_space = spaces.Box( np.array([-1.0] * self.max_action_dim), np.array([+1.0] * self.max_action_dim), dtype=np.float32 )
        else:
            self.action_space = spaces.Box(
                np.array([-1.0]*num_enabled_joints), np.array([+1.0]*num_enabled_joints), dtype=np.float32 )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        for k in self.bodies.keys():
            self.world.DestroyBody(self.bodies[k])
        self.bodies = {}
        self.joints = {}
        self.fixtures = {}

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

    def reset(self):
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

        # Process the fixtures
        # PolygonShape: vertices
        # EdgeShape: vertices
        # CircleShape: radius
        # friction, density, restitution, maskBits, categoryBits
        self.fixtures = {}
        for k in self.fixture_defs.keys():
            if self.fixture_defs[k]['FixtureShape']['Type'] == 'PolygonShape' or \
               self.fixture_defs[k]['FixtureShape']['Type'] == 'EdgeShape':
                fixture_shape = polygonShape(
                    vertices=[ (x/SCALE,y/SCALE)
                               for x,y in self.fixture_defs[k]['FixtureShape']['Vertices']])
            elif self.fixture_defs[k]['FixtureShape']['Type'] == 'CircleShape':
                fixture_shape = circleShape(
                    radius=self.fixture_defs[k]['FixtureShape']['Radius']/SCALE)
            else:
                print("Invalid fixture type: "+self.fixture_defs[k]['FixtureShape'])
                assert(False)
            self.fixtures[k] = fixtureDef(
                shape=fixture_shape,
                friction=self.fixture_defs[k]['Friction'],
                density=self.fixture_defs[k]['Density'],
                restitution=self.fixture_defs[k]['Restitution'],
                maskBits=self.fixture_defs[k]['MaskBits'],
                categoryBits=self.fixture_defs[k]['CategoryBits']
            )

        # Process the dynamic bodies
        # position, angle,  fixture,
        self.bodies = {}
        for k in self.body_defs.keys():
            if False and k == 'Hull':
                self.bodies[k] = self.world.CreateStaticBody(
                    position=[x/SCALE
                              for x in self.body_defs[k]['Position']],
                    angle=self.body_defs[k]['Angle'],
                    fixtures=[self.fixtures[f] for f in self.body_defs[k]['FixtureNames']]
                )
            else:
                self.bodies[k] = self.world.CreateDynamicBody(
                    position=[x/SCALE
                              for x in self.body_defs[k]['Position']],
                    angle=self.body_defs[k]['Angle'],
                    fixtures=[self.fixtures[f] for f in self.body_defs[k]['FixtureNames']]
                )
            self.bodies[k].color1 = self.body_defs[k]['Color1']
            self.bodies[k].color2 = self.body_defs[k]['Color2']
            self.bodies[k].can_touch_ground = self.body_defs[k]['CanTouchGround']
            self.bodies[k].ground_contact = False
            self.bodies[k].depth = self.body_defs[k]['Depth']
            self.bodies[k].connected_body   = []
            self.bodies[k].connected_joints = []

            # Apply a force to the 'center' body
            if k == 'Hull':
                self.bodies[k].ApplyForceToCenter(
                    (self.np_random.uniform(-self.body_defs[k]['InitialForceScale'], self.body_defs[k]['InitialForceScale']),
                     0), True)
        self.body_state_order = copy.deepcopy(list(self.bodies_to_report_keys))
        self.all_bodies_order = copy.deepcopy(self.body_state_order) + copy.deepcopy([k for k in self.body_defs.keys() if k not in self.body_state_order])
        for i in range(len(self.all_bodies_order)):
            k = self.all_bodies_order[i]
            self.bodies[k].index = i

        # Process the joint motors
        # bodyA, bodyB, localAnchorA, localAnchorB, enableMotor, enableLimit,
        # maxMotorTorque, motorSpeed, lowerAngle, upperAngle
        self.joints = {}
        for k in self.joint_defs.keys():
            self.joints[k] = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.bodies[self.joint_defs[k]['BodyA']],
                bodyB=self.bodies[self.joint_defs[k]['BodyB']],
                localAnchorA=[x/SCALE
                              for x in self.joint_defs[k]['LocalAnchorA']],
                localAnchorB=[x/SCALE
                              for x in self.joint_defs[k]['LocalAnchorB']],
                enableMotor=self.joint_defs[k]['EnableMotor'],
                enableLimit=self.joint_defs[k]['EnableLimit'],
                maxMotorTorque=self.joint_defs[k]['MaxMotorTorque'],
                motorSpeed=self.joint_defs[k]['MotorSpeed'],
                lowerAngle=self.joint_defs[k]['LowerAngle'],
                upperAngle=self.joint_defs[k]['UpperAngle']
            ))
            self.joints[k].depth = self.joint_defs[k]['Depth']
            self.joints[k].connected_body   = []
            self.joints[k].connected_joints = []

        # Process the body linkages
        # bodyA, bodyB, anchor
        self.linkages = {}
        for k in self.linkage_defs.keys():
            self.linkages[k] = self.world.CreateJoint(weldJointDef(
                bodyA=self.bodies[self.linkage_defs[k]['BodyA']],
                bodyB=self.bodies[self.linkage_defs[k]['BodyB']],
                localAnchorA=[x/SCALE for x in self.linkage_defs[k]['LocalAnchorA']],
                localAnchorB=[x/SCALE for x in self.linkage_defs[k]['LocalAnchorB']],
                frequencyHz=self.linkage_defs[k]['FrequencyHz']
            ))

        # Joints we want to report state is a superset of joints we want to allow action
        self.joint_action_order = copy.deepcopy(list(self.enabled_joints_keys))
        if self.truncate_state:
            self.joint_state_order = copy.deepcopy(self.joint_action_order) \
                + copy.deepcopy([k for k in self.joint_defs.keys()
                    if k not in self.joint_action_order and self.joint_defs[k]['ReportState']])
        else:
            self.joint_state_order = copy.deepcopy(self.joint_action_order) \
                + copy.deepcopy([k for k in self.joint_defs.keys()
                    if k not in self.joint_action_order])

        self.all_joints_order = copy.deepcopy(self.joint_state_order) + copy.deepcopy([k for k in self.joint_defs.keys() if k not in self.joint_state_order])
        for i in range(len(self.all_joints_order)):
            k = self.all_joints_order[i]
            self.joints[k].index = i

        # Construct index links between bodies and joints
        for k in self.joint_defs.keys():
            self.bodies[self.joint_defs[k]['BodyA']].connected_body.append(
                self.bodies[self.joint_defs[k]['BodyB']].index)
            self.bodies[self.joint_defs[k]['BodyB']].connected_body.append(
                self.bodies[self.joint_defs[k]['BodyA']].index)
            self.bodies[self.joint_defs[k]['BodyA']].connected_joints.append(
                self.joints[k].index)
            self.bodies[self.joint_defs[k]['BodyB']].connected_joints.append(
                self.joints[k].index)

            self.joints[k].connected_body.append(
                self.bodies[self.joint_defs[k]['BodyA']].index)
            self.joints[k].connected_body.append(
                self.bodies[self.joint_defs[k]['BodyB']].index)

        # Construct index links between hull segments via linkages
        for k in self.linkage_defs.keys():
            self.bodies[self.linkage_defs[k]['BodyA']].connected_body.append(
                self.bodies[self.linkage_defs[k]['BodyB']].index)
            self.bodies[self.linkage_defs[k]['BodyB']].connected_body.append(
                self.bodies[self.linkage_defs[k]['BodyA']].index)

        # Construct index links between joints connected to same body
        for k_joint in self.joints.keys():
            for i_body in self.joints[k_joint].connected_body:
                k_body = self.all_bodies_order[i_body]
                for i_jointB in self.bodies[k_body].connected_joints:
                    # Avoid adding self-links
                    if self.joints[k_joint].index == i_jointB:
                        continue
                    # Keep uniqueness of lists
                    if i_jointB in self.joints[k_joint].connected_joints:
                        continue
                    self.joints[k_joint].connected_joints.append(i_jointB)

        # Make sure hull is last
        self.drawlist = self.terrain + list(self.bodies.values())

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]

        ob, _, _, info = self.step(np.zeros(self.action_space.shape))
        return ob, info

    def _get_state(self):
        # State encoding:
        # For every body:
        #  angle, 2*angularVelocity/FPS, 0.3*velx*(VIEWPORT_W/SCALE)/FPS, 0.3*vely*(VIEWPORT_H/SCALE)/FPS, ground_contact
        # For every joint:
        #  angle, speed/SPEED
        state = []
        for i in range(len(self.body_state_order)):
            k = self.body_state_order[i]
            state += [
                self.bodies[k].angle,
                2.0*self.bodies[k].angularVelocity/FPS,
                0.3*self.bodies[k].linearVelocity.x*(VIEWPORT_W/SCALE)/FPS,
                0.3*self.bodies[k].linearVelocity.y*(VIEWPORT_H/SCALE)/FPS,
                1.0 if self.bodies[k].ground_contact else 0.0
            ]
        for i in range(len(self.joint_state_order)):
            k = self.joint_state_order[i]
            state += [
                self.joints[k].angle,
                self.joints[k].speed / self.joint_defs[k]['Speed'],
            ]

        # Zero-pad state if max_state_dim is larger than current state
        if self.max_state_dim is not None:
            while (len(state) < self.max_state_dim - 10):
                state.append(0.0)

        state += [l.fraction for l in self.lidar]

        if self.max_state_dim is not None:
            assert len(state) == self.max_state_dim
        else:
            assert len(state)==(5*len(self.body_state_order)+2*len(self.joint_state_order)+10)

        return state

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        #control_speed = True
        if control_speed:
            for a in range(len(self.joint_action_order)):
                k = self.joint_action_order[a]
                self.joints[k].motorSpeed = float(self.joint_defs[k]['Speed'] * np.clip(action[a], -1, 1))
        else:
            for a in range(len(self.joint_action_order)):
                k = self.joint_action_order[a]
                self.joints[k].motorSpeed = float(self.joint_defs[k]['Speed'] * np.sign(action[a]))
                self.joints[k].maxMotorTorque = float(self.joint_defs[k]['MaxMotorTorque'] * np.clip(np.abs(action[a]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.bodies['Hull'].position
        vel = self.bodies['Hull'].linearVelocity

        # Cache lidar results
        for i in range(10):
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

        for i, a in enumerate(action):
            # Don't discount for padded action space
            if i >= len(self.joint_action_order):
                continue
            reward -= 0.00035 * self.joint_defs[self.joint_action_order[i]]['MaxMotorTorque'] * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, self._info_dict()

    def _info_dict(self):
                # Construct the info dict so that the grid state space and action space can be built
        info = {}
        info['hull_x'] = self.bodies['Hull'].position.x
        info['hull_y'] = self.bodies['Hull'].position.y
        info['bodies'] = []
        for i in range(len(self.body_state_order)):
            k = self.body_state_order[i]
            b = self.bodies[k]
            info['bodies'].append((
                b.position.x, b.position.y, b.angle,
                2.0*b.angularVelocity/FPS,
                0.3*b.linearVelocity.x*(VIEWPORT_W/SCALE)/FPS,
                0.3*b.linearVelocity.y*(VIEWPORT_H/SCALE)/FPS,
                1.0 if b.ground_contact else 0,
                # Depth: 0.0 if "front", 1.0 if "back", hull is considered front and second leg is back leg
                b.depth,
                b.connected_body,
                b.connected_joints
            ))
        info['joints'] = []
        for i in range(len(self.joint_state_order)):
            k = self.joint_state_order[i]
            j = self.joints[k]
            info['joints'].append((
                j.anchorA.x, j.anchorA.y, j.anchorB.x, j.anchorB.y,
                j.angle,
                j.speed / self.joint_defs[k]['Speed'],
                j.depth,
                j.connected_body,
                j.connected_joints
            ))
        info['lidar'] = self.lidar
        return info

    def coord_to_grid(self, coord, zero):
        return round((coord - zero) / self.grid_scale * self.grid_edge)

    def get_zeros(self):
        return self.bodies['Hull'].position.x - self.grid_scale / 2, self.bodies['Hull'].position.y - self.grid_scale / 2

    def _draw_stategrid(self, model, alpha=0.5):
        self.grid_edge = model.senc_nngrid.grid_edge
        self.grid_scale = model.senc_nngrid.grid_scale
        self.grid_square_edge = self.grid_scale / self.grid_edge

        # Draw state grid around hull and highlight squares with a nonzero channel
        fill_empty = True
        show_grid = True

        zero_x, zero_y = self.get_zeros()
        filled_in_squares = []

        body_color = (1, 1, 1, alpha)
        joint_color = (1, 1, 1, alpha)

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

    # Actiongrid modes are 'gray' for grayscale, 'heat' for reddish gradient,
    # and 'rainbow' for all colors
    def _draw_actiongrid(self, model, depth=-1, clip_values=True, alpha=0.5, mode='gray'):
        # Dimensions of action grid output by model not always the same as those of the state grid
        self.grid_edge = model.adec_nngrid.current_actiongrid.shape[2]
        self.grid_scale = model.adec_nngrid.grid_scale
        self.grid_square_edge = self.grid_scale / self.grid_edge

        # Draw action grid around hull and color in grayscale intensity of action
        show_grid = True

        zero_x, zero_y = self.get_zeros()

        # Depth of -1 means flatten depth layers and display
        if depth >= 0:
            current_actiongrid_layer = model.adec_nngrid.current_actiongrid[0, depth]
        else:
            current_actiongrid_layer = np.sum(model.adec_nngrid.current_actiongrid, axis=1)[0]
        if clip_values:
            min_sum = 0.0
            max_sum = np.amax(current_actiongrid_layer)
            fill_range = max_sum
        else:
            min_sum = np.amin(current_actiongrid_layer)
            max_sum = np.amax(current_actiongrid_layer)
            fill_range = max_sum - min_sum

        for x in range(self.grid_edge):
            for y in range(self.grid_edge):
                lower_left_x, lower_left_y = zero_x + self.grid_square_edge * x, zero_y + self.grid_square_edge * y
                vertices = [(lower_left_x, lower_left_y),
                    (lower_left_x + self.grid_square_edge, lower_left_y),
                    (lower_left_x + self.grid_square_edge, lower_left_y + self.grid_square_edge),
                    (lower_left_x, lower_left_y + self.grid_square_edge)]
                filled_in_square = self.viewer.draw_polygon(vertices)
                # Set intensity to the relative value of channels
                current_value = current_actiongrid_layer[x, y]
                if clip_values and current_value < 0.0:
                    current_value = 0.0
                square_proportion = (current_value - min_sum) / fill_range
                square_red = 0.0
                square_green = 0.0
                square_blue = 0.0
                if mode == 'gray':
                    square_red = square_proportion
                    square_green = square_proportion
                    square_blue = square_proportion
                elif mode == 'heat':
                    # First increase red, then green, then blue
                    if square_proportion < 0.333:
                        square_red = square_proportion / 0.333
                    elif square_proportion < 0.667:
                        square_red = 1.0
                        square_green = (square_proportion - 0.333) / 0.333
                    else:
                        square_red = 1.0
                        square_green = 1.0
                        square_blue = (square_proportion - 0.667) / 0.333
                elif mode == 'rainbow':
                    # Interpolate hue between 0 (red) and 240 (blue)
                    hue = (1.0 - square_proportion) * 0.667
                    square_red, square_green, square_blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

                filled_in_square._color.vec4 = (square_red,
                    square_green, square_blue, alpha)

        if show_grid:
            for i in range(self.grid_edge + 1):
                vertical = [(zero_x + self.grid_square_edge * i, zero_y), (zero_x + self.grid_square_edge * i, zero_y + self.grid_scale)]
                horizontal = [(zero_x, zero_y + self.grid_square_edge * i), (zero_x + self.grid_scale, zero_y + self.grid_square_edge * i)]
                self.viewer.draw_polyline(vertical, color=(0, 0, 0), linewidth=1)
                self.viewer.draw_polyline(horizontal, color=(0, 0, 0), linewidth=1)



    def render(self, mode='human', model=None, show_stategrid=False, actiongrid_mode='hide', actiongrid_depth=-1, actiongrid_clip=True, alpha=0.5):
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
            self._draw_stategrid(model, alpha=alpha)
        if actiongrid_mode != 'hide':
            self._draw_actiongrid(model, depth=actiongrid_depth,
                clip_values=actiongrid_clip, mode=actiongrid_mode, alpha=alpha)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class JSONWalkerHardcore(JSONWalker):
    hardcore = True

if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    #env = JSONWalker("box2d-json/BipedalWalker.json")
    #env = JSONWalker('box2d-json/HumanoidWalker.json')
    #env = JSONWalker('box2d-json/HumanoidFeetWalker.json')
    #env = JSONWalker('box2d-json/RaptorWalker.json')
    #env = JSONWalkerHardcore('box2d-json/DogWalker.json')
    #env = JSONWalker('box2d-json/CentipedeWalker.json')
    #env = JSONWalker('box2d-json-gen-bipedal-segments-baseline/train/GeneratedBipedalWalker0.json')
    #env = JSONWalker('box2d-json-gen/GeneratedBipedalWalker.json')
    #env = JSONWalker('datasets/bipedal-random-offcenter-hull-1-12-25-percent/train/GeneratedBipedalWalker4segments-4.json')
    #env = JSONWalker('box2d-json-gen/GeneratedCentipedeWalker.json')
    env = JSONWalker(jsonfile='box2d-json-gen/GeneratedRaptorWalker.json')

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
        obs, r, done, info = env.step(a)
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
