import sys, math, json, copy
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

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

    def __init__(self, jsonfile):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None

        self.prev_shaping = None

        # JSON loading
        with open(jsonfile) as f:
            self.jsondata = json.load(f)

        self.fixture_defs = {}
        self.body_defs    = {}
        self.joint_defs   = {}

        # Split the data so that we can create it later in order
        for k in self.jsondata.keys():
            if self.jsondata[k]['DataType'] == 'Fixture':
                self.fixture_defs[k] = copy.deepcopy(self.jsondata[k])
            elif self.jsondata[k]['DataType'] == 'DynamicBody':
                self.body_defs[k] = copy.deepcopy(self.jsondata[k])
            elif self.jsondata[k]['DataType'] == 'JointMotor':
                self.joint_defs[k] = copy.deepcopy(self.jsondata[k])
            else:
                assert(False)


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

        num_joints = len(self.joint_defs.keys())
        high = np.array( [np.inf]*(5*len(self.body_defs)+2*len(self.joint_defs)+10) )

        self.action_space = spaces.Box(np.array([-1]*num_joints), np.array([+1]*num_joints))
        self.observation_space = spaces.Box(-high, high)

        self.reset()

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
            self.bodies[k] = self.world.CreateDynamicBody(
                position=[x/SCALE for x in self.body_defs[k]['Position']],
                angle=self.body_defs[k]['Angle'],
                fixtures=self.fixtures[self.body_defs[k]['FixtureName']]
            )
            self.bodies[k].color1 = self.body_defs[k]['Color1']
            self.bodies[k].color2 = self.body_defs[k]['Color2']
            self.bodies[k].can_touch_ground = self.body_defs[k]['CanTouchGround']
            self.bodies[k].ground_contact = False

            # Apply a force to the 'center' body
            if k == 'Hull':
                self.bodies[k].ApplyForceToCenter(
                    (self.np_random.uniform(-self.body_defs[k]['InitialForceScale'], self.body_defs[k]['InitialForceScale']), 
                     0), True)
        self.body_state_order = copy.deepcopy(list(self.bodies.keys()))

        # Process the joint motors
        # bodyA, bodyB, localAnchorA, localAnchorB, enableMotor, enableLimit,
        # maxMotorTorque, motorSpeed, lowerAngle, upperAngle
        self.joints = {}
        for k in self.joint_defs.keys():
            self.joints[k] = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.bodies[self.joint_defs[k]['BodyA']],
                bodyB=self.bodies[self.joint_defs[k]['BodyB']],
                localAnchorA=[x/SCALE for x in self.joint_defs[k]['LocalAnchorA']],
                localAnchorB=[x/SCALE for x in self.joint_defs[k]['LocalAnchorB']],
                enableMotor=self.joint_defs[k]['EnableMotor'],
                enableLimit=self.joint_defs[k]['EnableLimit'],
                maxMotorTorque=self.joint_defs[k]['MaxMotorTorque'],
                motorSpeed=self.joint_defs[k]['MotorSpeed'],
                lowerAngle=self.joint_defs[k]['LowerAngle'],
                upperAngle=self.joint_defs[k]['UpperAngle']
            ))
        self.joint_action_order = copy.deepcopy(list(self.joints.keys()))

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

        return self.step(np.array([0]*np.prod(self.action_space.shape)))[0]

    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
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

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

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
        for i in range(len(self.joint_action_order)):
            k = self.joint_action_order[i]
            state += [
                self.joints[k].angle,
                self.joints[k].speed / self.joint_defs[k]['Speed'],
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==(5*len(self.body_state_order)+2*len(self.joint_action_order)+10)

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for i, a in enumerate(action):
            reward -= 0.00035 * self.joint_defs[self.joint_action_order[i]]['MaxMotorTorque'] * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def render(self, mode='human'):
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
    env = JSONWalker('box2d-json/HumanoidFeetWalker.json')
    env.reset()
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
        s, r, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
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
