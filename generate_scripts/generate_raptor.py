# TODO(josh): can simplify script by not making distinction between 'Head', 'Neck', 'Tail', and
# 'Hull', make them all 'Body' but name the center body 'Hull' for json_walker.py purposes
import sys
import os
import math
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rigid-spine', dest='rigid_spine', action='store_true')
    parser.add_argument('--no-rigid-spine', dest='rigid_spine', action='store_false')
    parser.set_defaults(rigid_spine=False)
    parser.add_argument('--spine-motors', dest='spine_motors', action='store_true')
    parser.add_argument('--no-spine-motors', dest='spine_motors', action='store_false')
    parser.set_defaults(spine_motors=True)
    # Add option to randomize density for each body part, or interpolate density on neck/tail
    # Can also change restitution?
    parser.add_argument(
        '--hull-density',
        type=float,
        default=5.0,
        help='The density of the center hull segment (default 5.0)')
    parser.add_argument(
        '--hull-friction',
        type=float,
        default=0.1,
        help='The friction of the center hull segment (default 0.1)')
    parser.add_argument(
        '--head-density',
        type=float,
        default=5.0,
        help='The density of the head (default 5.0)')
    parser.add_argument(
        '--head-friction',
        type=float,
        default=0.0,
        help='The friction of the head (default 0.0)')
    parser.add_argument(
        '--neck-density',
        type=float,
        default=5.0,
        help='The density of neck segments (default 5.0)')
    parser.add_argument(
        '--neck-friction',
        type=float,
        default=0.0,
        help='The friction of neck segments (default 0.0)')
    parser.add_argument(
        '--tail-density',
        type=float,
        default=5.0,
        help='The density of tail segments (default 5.0)')
    parser.add_argument(
        '--tail-friction',
        type=float,
        default=0.0,
        help='The friction of tail segments (default 28.0)')
    parser.add_argument(
        '--leg-density',
        type=float,
        default=1.0,
        help='The density of leg segments (default 1.0)')
    parser.add_argument(
        '--leg-friction',
        type=float,
        default=0.2,
        help='The friction of leg segments (default 0.2)')

    parser.add_argument(
        '--neck-segments',
        type=int,
        default=4,
        help='Number of neck segments (default 4)')
    parser.add_argument(
        '--tail-segments',
        type=int,
        default=4,
        help='Number of tail segments (default 4)')
    parser.add_argument(
        '--filename',
        type=str,
        default='box2d-json-gen/GeneratedRaptorWalker.json',
        help='What to call the output JSON file')
    parser.add_argument(
        '--hull-width',
        type=float,
        default=28.0,
        help='The width of the center hull segment (default 28.0)')
    parser.add_argument(
        '--hull-height',
        type=float,
        default=28.0,
        help='The height of the center hull segment (default 28.0)')
    parser.add_argument(
        '--head-width',
        type=float,
        default=12.5,
        help='Head width (default 12.5)')
    parser.add_argument(
        '--head-height',
        type=float,
        default=24.0,
        help='Head height (default 24.0)')
    parser.add_argument(
        '--head-angle',
        type=float,
        default=0.52,
        help='Head angle (default 0.52)')
    parser.add_argument(
        '--neck-width',
        type=float,
        default=9.0,
        help='Ending neck width, right before head (default 9.0)')
    parser.add_argument(
        '--neck-height',
        type=float,
        default=9.0,
        help='Ending neck height, right before head (default 9.0)')
    parser.add_argument(
        '--tail-width',
        type=float,
        default=22.0,
        help='Ending tail width (default 22.0)')
    parser.add_argument(
        '--tail-height',
        type=float,
        default=4.0,
        help='Ending tail height (default 4.0)')
    parser.add_argument(
        '--thigh-width',
        type=float,
        default=10.0,
        help='Thigh (highest up leg segment) width (default 10.0)')
    parser.add_argument(
        '--thigh-height',
        type=float,
        default=34.0,
        help='Thigh (highest up leg segment) height (default 34.0)')
    parser.add_argument(
        '--shin-width',
        type=float,
        default=7.0,
        help='Shin (second highest leg segment) width (default 7.0)')
    parser.add_argument(
        '--shin-height',
        type=float,
        default=24.0,
        help='Shin (second highest leg segment) height (default 24.0)')
    parser.add_argument(
        '--foot-width',
        type=float,
        default=6.4,
        help='Foot (third highest leg segment) width (default 6.4)')
    parser.add_argument(
        '--foot-height',
        type=float,
        default=20.0,
        help='Foot (third highest leg segment) height (default 20.0)')
    parser.add_argument(
        '--toes-width',
        type=float,
        default=5.6,
        help='Toes (bottom leg segment) width (default 5.6)')
    parser.add_argument(
        '--toes-height',
        type=float,
        default=16.0,
        help='Toes (bottom leg segment) height (default 16.0)')
    return vars(parser.parse_args())

LIGHT_COLOR = [0.6392156862745098, 0.6941176470588235, 0.7372549019607844]
MID_COLOR = [0.47058823529411764, 0.5490196078431373, 0.6862745098039216]
DARK_COLOR = [0.35294117647058826, 0.4117647058823529, 0.47058823529411764]
LINE_COLOR = [0.23921568627450981, 0.2823529411764706, 0.35294117647058826]

# ith_between indexes intermediate segments as [1...total] where i=0 is start, not the first segment between, and i=total is end
def ith_between(start, end, i, total):
    interval = (end - start) / total
    return interval * i + start

class GenerateRaptor:
    def __init__(self, args):
        self.args = args
        self.output = {}
        # Red flag is at about x=50
        self.start_x = 50
        for i in range(args['tail_segments']):
            self.start_x += ith_between(
                self.args['hull_width'],
                self.args['tail_width'],
                i+1,
                self.args['tail_segments']
            )
        self.start_x += 0.5 * self.args['hull_width']
        # Ground starts at y=100
        self.start_y = 100
        # Currently, naively calculate total leg height
        for k in ['thigh', 'foot']:
            self.start_y += self.args[k + '_height']
        for k in ['shin', 'toes']:
            self.start_y += self.args[k + '_width']

    def build(self):
        self.build_fixtures()

        self.build_bodies()

        self.build_joints()

    def build_fixtures(self):
        self.output['HullFixture'] = {}
        self.output['HeadFixture'] = {}

        leg_fixtures = ['ThighFixture', 'ShinFixture', 'FootFixture', 'ToesFixture']

        for f in leg_fixtures:
            self.output[f] = {}

        # Build neck and tail separately
        start_width, start_height = self.args['hull_width'], self.args['hull_height']
        end_width, end_height = self.args['neck_width'], self.args['neck_height']

        for i in range(self.args['neck_segments']):
            k = 'Neck' + str(i) + 'Fixture'
            self.output[k] = {}
            self.output[k]['DataType'] = 'Fixture'
            self.output[k]['FixtureShape'] = {}
            self.output[k]['FixtureShape']['Type'] = 'PolygonShape'
            half_width = 0.5 * ith_between(
                start_width,
                end_width,
                i+1,
                self.args['neck_segments']
            )
            half_height = 0.5 * ith_between(
                start_height,
                end_height,
                i+1,
                self.args['neck_segments']
            )
            self.output[k]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]

        start_width, start_height = self.args['hull_width'], self.args['hull_height']
        end_width, end_height = self.args['tail_width'], self.args['tail_height']

        for i in range(self.args['tail_segments']):
            k = 'Tail' + str(i) + 'Fixture'
            self.output[k] = {}
            self.output[k]['DataType'] = 'Fixture'
            self.output[k]['FixtureShape'] = {}
            self.output[k]['FixtureShape']['Type'] = 'PolygonShape'
            half_width = 0.5 * ith_between(
                start_width,
                end_width,
                i+1,
                self.args['tail_segments']
            )
            half_height = 0.5 * ith_between(
                start_height,
                end_height,
                i+1,
                self.args['tail_segments']
            )
            self.output[k]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]

        for f in self.output.keys():
            if not 'Neck' in f and not 'Tail' in f:
                self.output[f]['DataType'] = 'Fixture'
                self.output[f]['FixtureShape'] = {}
                self.output[f]['FixtureShape']['Type'] = 'PolygonShape'

                prefix = f.split('Fixture')[0].lower()

                half_width, half_height = 0.5 * self.args[prefix + '_width'], 0.5 * self.args[prefix + '_height']

                self.output[f]['FixtureShape']['Vertices'] = [
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height]
                ]

            if 'Hull' in f:
                prefix = 'hull'
            elif f in leg_fixtures:
                prefix = 'leg'
            elif 'Neck' in f:
                prefix = 'neck'
            elif 'tail' in f:
                prefix = 'tail'
            self.output[f]['Friction'] = self.args[prefix + '_friction']
            self.output[f]['Density'] = self.args[prefix + '_density']
            self.output[f]['Restitution'] = 0.0
            self.output[f]['MaskBits'] = 1
            self.output[f]['CategoryBits'] = 32

    def body_add_position_angle_neck_or_tail(self, neck_or_tail):
        title_neck_or_tail = neck_or_tail.title()
        # Whether to build in the positive direction (neck) or negative (tail)
        x_dir = 1.0 if neck_or_tail == 'neck' else -1.0
        # Fill in position + angle for neck and head
        if self.args[neck_or_tail + '_segments'] > 0:
            current_x = self.start_x + x_dir * (0.5 * self.args['hull_width'] + 0.5 * ith_between(
                self.args['hull_width'],
                self.args[neck_or_tail + '_width'],
                1,
                self.args[neck_or_tail + '_segments']
            ))
            # Align segments along top spine
            current_y = self.start_y + 0.5 * (self.args['hull_height'] - ith_between(
                self.args['hull_height'],
                self.args[neck_or_tail + '_height'],
                1,
                self.args[neck_or_tail + '_segments']
            ))
        elif neck_or_tail == 'neck':
            head_x = self.start_x + 0.5 * self.args['hull_width'] + 0.5 * self.args['head_width']
            head_y = self.start_y + 0.5 * (self.args['hull_height'] - self.args['head_height'])
            self.output['Head']['Angle'] = 0.0
            self.output['Head']['Position'] = [head_x, head_y]
            return
        else:
            return

        for i in range(self.args[neck_or_tail + '_segments']):
            self.output[title_neck_or_tail + str(i)]['Angle'] = 0
            #ith_between(0.25, 0.65, i, self.args[neck_or_tail + '_segments'])
            # TODO(josh): make this support a different position based on angle
            self.output[title_neck_or_tail + str(i)]['Position'] = [current_x, current_y]

            # Set up current_x and current_y for building head if last of neck
            if neck_or_tail == 'neck' and i == self.args[neck_or_tail + '_segments'] - 1:
                #
                current_x = current_x + ith_between(
                    self.args['hull_width'],
                    self.args['neck_width'],
                    i+1,
                    self.args['neck_segments']
                ) + 0.5 * self.args['head_width']
                current_y = current_y + 0.25 * (ith_between(
                    self.args['hull_height'],
                    self.args['neck_height'],
                    i+1,
                    self.args['neck_segments']
                ) - self.args['head_height'])

                self.output['Head']['Angle'] = self.args['head_angle']
                self.output['Head']['Position'] = [current_x, current_y]
            else:
                current_x = current_x + x_dir * (0.5 * ith_between(
                    self.args['hull_width'],
                    self.args[neck_or_tail + '_width'],
                    i+1,
                    self.args[neck_or_tail + '_segments']
                ) + 0.5 * ith_between(
                    self.args['hull_width'],
                    self.args[neck_or_tail + '_width'],
                    i+2,
                    self.args[neck_or_tail + '_segments']
                ))
                current_y = current_y + 0.5 * (ith_between(
                    self.args['hull_height'],
                    self.args[neck_or_tail + '_height'],
                    i+1,
                    self.args[neck_or_tail + '_segments']
                ) - ith_between(
                    self.args['hull_height'],
                    self.args[neck_or_tail + '_height'],
                    i+2,
                    self.args[neck_or_tail + '_segments']
                ))

    def build_leg_bodies(self):
        for f in ['ThighFixture', 'ShinFixture', 'FootFixture', 'ToesFixture']:
            for sign in [-1, +1]:
                k = f.split('Fixture')[0] + str(sign)
                self.output[k] = {}
                self.output[k]['DataType'] = 'DynamicBody'
                self.output[k]['FixtureNames'] = [f]
                self.output[k]['Color1'] = DARK_COLOR if '-1' in k else MID_COLOR
                self.output[k]['Color2'] = LINE_COLOR
                self.output[k]['CanTouchGround'] = True
                self.output[k]['InitialForceScale'] =  0
                self.output[k]['Depth'] = 0 if '-1' in k else -1

                # Add position and angle
                # TODO(josh): make angle of legs an argument?
                # TODO(josh): make foot and toes not rotated?
                thigh_x = self.start_x + math.sin(0.3) * 0.5 * self.args['thigh_height']
                thigh_y = self.start_y - math.cos(0.3) * 0.5 * self.args['thigh_height']
                shin_x = thigh_x - 0.5 * self.args['shin_height']
                shin_y = thigh_y - 0.5 * self.args['thigh_height']
                foot_x = shin_x - 0.5 * self.args['shin_height']
                foot_y = shin_y - 0.5 * self.args['foot_height']
                toes_x = foot_x + 0.5 * self.args['toes_height']
                toes_y = foot_y - 0.5 * self.args['foot_height']
                if 'Thigh' in k:
                    self.output[k]['Position'] = [thigh_x, thigh_y]
                    self.output[k]['Angle'] = 0.3
                elif 'Shin' in k:
                    # Note: shin is sideways
                    self.output[k]['Position'] = [shin_x, shin_y]
                    self.output[k]['Angle'] = -1.2
                elif 'Foot' in k:
                    self.output[k]['Position'] = [foot_x, foot_y]
                    self.output[k]['Angle'] = 0.25
                elif 'Toes' in k:
                    # Toes is also sideways
                    self.output[k]['Position'] = [toes_x, toes_y]
                    self.output[k]['Angle'] = 1.45

    def build_bodies(self):
        start_x = self.start_x
        start_y = self.start_y

        # Build common features (but build legs separately)
        fixtures = list(self.output.keys())
        for k in fixtures:
            body_name = k.split('Fixture')[0]

            if body_name in ['Thigh', 'Shin', 'Foot', 'Toes']:
                continue

            self.output[body_name] = {}
            self.output[body_name]['DataType'] = 'DynamicBody'
            self.output[body_name]['FixtureNames'] = [k]
            self.output[body_name]['Color1'] = MID_COLOR
            self.output[body_name]['Color2'] = LINE_COLOR
            self.output[body_name]['CanTouchGround'] = 'Tail' in body_name
            self.output[body_name]['InitialForceScale'] = 100 if body_name == 'Hull' else 0
            self.output[body_name]['Depth'] = 0

        # Fill in position + angle for hull
        self.output['Hull']['Position'] = [start_x, start_y]
        self.output['Hull']['Angle'] = 0

        # Fill in position + angle neck and tail
        self.body_add_position_angle_neck_or_tail('neck')
        self.body_add_position_angle_neck_or_tail('tail')

        # Build bodies for 2 legs
        self.build_leg_bodies()

    def build_neck_or_tail_joints(self, neck_or_tail):
        title_neck_or_tail = neck_or_tail.title()
        x_dir = 1.0 if neck_or_tail == 'neck' else -1.0
        joint_counter = 0 if neck_or_tail == 'neck' else self.args['neck_segments'] + 1

        if neck_or_tail == 'neck' and self.args['neck_segments'] == 0:
            k = 'Joint' + str(joint_counter) + '.Hull.Head'
            self.output[k] = {}
            self.output[k]['BodyA'] = 'Hull'
            self.output[k]['BodyB'] = 'Head'

            if self.args['rigid_spine']:
                self.output[k]['Anchor'] = [
                    self.start_x + 0.5 * self.args['hull_width'],
                    self.start_y + 0.25 * self.args['hull_height']
                ]
            else:
                self.output[k]['LocalAnchorA'] = [0.5 * self.args['hull_width'], 0]
                self.output[k]['LocalAnchorB'] = [
                    -0.5 * self.args['head_width'],
                    0.25 * self.args['head_height']
                ]
                # Use neck upper/shin angles for this edge case
                self.output[k]['LowerAngle'] = -0.5
                self.output[k]['UpperAngle'] = 0.2
            return
        elif self.args[neck_or_tail + '_segments'] == 0:
            return

        k = 'Joint' + str(joint_counter) + '.Hull.' + title_neck_or_tail + '0'
        self.output[k] = {}
        self.output[k]['BodyA'] = 'Hull'
        self.output[k]['BodyB'] = title_neck_or_tail + '0'
        current_width = ith_between(
            self.args['hull_width'],
            self.args[neck_or_tail + '_width'],
            1,
            self.args[neck_or_tail + '_segments']
        )
        current_height = ith_between(
            self.args['hull_height'],
            self.args[neck_or_tail + '_height'],
            1,
            self.args[neck_or_tail + '_segments']
        )
        current_x = self.start_x + x_dir * 0.5 * self.args['hull_width']
        current_y = self.start_y + 0.5 * (self.args['hull_height'] - current_height)
        if self.args['rigid_spine']:
            self.output[k]['Anchor'] = [current_x, current_y]
        else:
            self.output[k]['LocalAnchorA'] = [
                x_dir * 0.5 * self.args['hull_width'],
                0.5 * (self.args['hull_height'] - current_height)
            ]
            self.output[k]['LocalAnchorB'] = [-x_dir * 0.5 * current_width, 0]
            self.output[k]['LowerAngle'] = -0.5
            self.output[k]['UpperAngle'] = 0.2
        joint_counter += 1

        for i in range(self.args[neck_or_tail + '_segments'] - 1):
            k = 'Joint' + str(joint_counter) + '.' + title_neck_or_tail + str(i) + '.' + title_neck_or_tail + str(i+1)
            prev_width = current_width
            prev_height = current_height
            current_width= ith_between(
                self.args['hull_width'],
                self.args[neck_or_tail + '_width'],
                i+2,
                self.args[neck_or_tail + '_segments']
            )
            current_height = ith_between(
                self.args['hull_height'],
                self.args[neck_or_tail + '_height'],
                i+2,
                self.args[neck_or_tail + '_segments']
            )
            current_x += x_dir * prev_width
            current_y += 0.5 * (prev_height - current_height)
            self.output[k] = {}
            self.output[k]['BodyA'] = title_neck_or_tail + str(i)
            self.output[k]['BodyB'] = title_neck_or_tail + str(i+1)
            if self.args['rigid_spine']:
                self.output[k]['Anchor'] = [current_x, current_y]
            else:
                self.output[k]['LocalAnchorA'] = [
                    x_dir * 0.5 * prev_width,
                    0.5 * (prev_height - current_height)
                ]
                self.output[k]['LocalAnchorB'] = [-x_dir * 0.5 * current_width, 0]
                self.output[k]['LowerAngle'] = -0.5
                self.output[k]['UpperAngle'] = 0.2
            joint_counter += 1

        # If building neck joints, build head joint
        if neck_or_tail == 'neck':
            # Head-neck joint
            k = 'Joint' + str(joint_counter) + '.Neck' + str(self.args['neck_segments'] - 1) + '.Head'
            self.output[k] = {}
            self.output[k]['BodyA'] = 'Neck' + str(self.args['neck_segments'] - 1)
            self.output[k]['BodyB'] = 'Head'
            current_width = ith_between(
                self.args['hull_width'],
                self.args['neck_width'],
                self.args['neck_segments'],
                self.args['neck_segments']
            )
            current_x += current_width
            if self.args['rigid_spine']:
                self.output[k]['Anchor'] = [current_x, current_y]
            else:
                self.output[k]['LocalAnchorA'] = [x_dir * 0.5 * current_width, 0]
                self.output[k]['LocalAnchorB'] = [
                    -0.5 * self.args['head_width'],
                    0.25 * self.args['head_height']
                ]
                self.output[k]['LowerAngle'] = -0.9
                self.output[k]['UpperAngle'] = 0.7
            joint_counter += 1

    def build_leg_joints(self):
        joint_counter = self.args['neck_segments'] + self.args['tail_segments'] + 1
        leg_names = ['Thigh', 'Shin', 'Foot', 'Toes']

        for sign in [-1, +1]:
            k = 'Joint' + str(joint_counter) + '.Hull.Thigh' + str(sign)
            self.output[k] = {}
            self.output[k]['BodyA'] = 'Hull'
            self.output[k]['BodyB'] = 'Thigh' + str(sign)
            self.output[k]['LocalAnchorA'] = [0.0, 0.0]
            self.output[k]['LocalAnchorB'] = [0.0, 0.5 * self.args['thigh_height']]
            self.output[k]['LowerAngle'] = -0.8
            self.output[k]['UpperAngle'] = 1.1

            joint_counter += 1

            for i in range(len(leg_names)-1):
                k = 'Joint' + str(joint_counter) + '.' + leg_names[i] + str(sign) + '.' + leg_names[i+1] + str(sign)
                self.output[k] = {}
                self.output[k]['BodyA'] = leg_names[i] + str(sign)
                self.output[k]['BodyB'] = leg_names[i+1] + str(sign)
                self.output[k]['LocalAnchorA'] = [0.0, -0.5 * self.args[leg_names[i].lower() + '_height']]
                self.output[k]['LocalAnchorB'] = [0.0, 0.5 * self.args[leg_names[i+1].lower() + '_height']]
                self.output[k]['LowerAngle'] = -0.8
                if leg_names[i] == 'Thigh':
                    self.output[k]['UpperAngle'] = 1.1
                elif leg_names[i] == 'Shin':
                    self.output[k]['UpperAngle'] = 0.5
                elif leg_names[i] == 'Foot':
                    self.output[k]['UpperAngle'] = 0.8

                self.output[k]['Depth'] = 0 if sign == -1 else 1

                joint_counter += 1

    def build_joints(self):
        # Build neck and tail joints
        self.build_neck_or_tail_joints('neck')
        self.build_neck_or_tail_joints('tail')

        # Build leg joints
        self.build_leg_joints()

        body_names = ['Hull', 'Head', 'Neck', 'Tail']
        # Write common parts of joints
        for k in self.output.keys():
            if 'Joint' in k:
                first_body_name = k.split('.')[1]
                second_body_name = k.split('.')[2]

                is_body_joint = any(name in first_body_name for name in body_names) and any(name in second_body_name for name in body_names)
                if is_body_joint and self.args['rigid_spine']:
                    self.output[k]['DataType'] = 'Linkage'
                    self.output[k]['Depth'] = 0 if '-1' in k else 1
                else:
                    self.output[k]['DataType'] = 'JointMotor'
                    self.output[k]['EnableMotor'] = self.args['spine_motors'] or not is_body_joint
                    self.output[k]['EnableLimit'] = True
                    self.output[k]['MaxMotorTorque'] = 80
                    self.output[k]['MotorSpeed'] = 0.0
                    self.output[k]['Speed'] = 1
                    self.output[k]['Depth'] = 0 if '-1' in k else 1

    def write_to_json(self, filename=None):
        if filename is None:
            outfile = open(self.args['filename'], 'w+')
        else:
            outfile = open(filename, 'w+')

        outfile.write(json.dumps(self.output, indent=4, separators=(',', ' : ')))


if __name__ == '__main__':
    args = parse_args()

    gen = GenerateRaptor(args)

    gen.build()

    gen.write_to_json()

