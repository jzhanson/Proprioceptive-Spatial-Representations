import sys
import os
import math
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

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
        '--filename',
        type=str,
        default='box2d-json-gen/GeneratedBipedalWalker.json',
        help='What to call the output JSON file')
    parser.add_argument(
        '--body-segments',
        type=int,
        default=1,
        help='How many body segments to split the hull into')
    parser.add_argument('--rigid-spine', dest='rigid_spine', action='store_true')
    parser.add_argument('--no-rigid-spine', dest='rigid_spine', action='store_false')
    parser.set_defaults(rigid_spine=False)
    parser.add_argument('--spine-motors', dest='spine_motors', action='store_true')
    parser.add_argument('--no-spine-motors', dest='spine_motors', action='store_false')
    parser.set_defaults(spine_motors=True)
    parser.add_argument(
        '--hull-width',
        type=float,
        default=64.0,
        help='The width of the center hull segment (default 64.0)')
    parser.add_argument(
        '--hull-height',
        type=float,
        default=16.0,
        help='The height of the center hull segment (default 28.0)')
    parser.add_argument(
        '--leg-width',
        type=float,
        default=8.0,
        help='Leg (highest up leg segment) width (default 8.0)')
    parser.add_argument(
        '--leg-height',
        type=float,
        default=34.0,
        help='Leg (highest up leg segment) height (default 34.0)')
    parser.add_argument(
        '--lower-width',
        type=float,
        default=6.4,
        help='Lower (second leg segment) width (default 6.4)')
    parser.add_argument(
        '--lower-height',
        type=float,
        default=34.0,
        help='Lower (second leg segment) height (default 34.0)')
    return vars(parser.parse_args())

LIGHT_COLOR = [0.4, 0.2, 0.4]
HULL_COLOR = [0.5, 0.4, 0.9]
DARK_COLOR = [0.7, 0.4, 0.6]
LINE_COLOR = [0.3, 0.3, 0.5]

# ith_between indexes intermediate segments as [1...total] where i=0 is start, not the first segment between, and i=total is end
def ith_between(start, end, i, total):
    interval = (end - start) / total
    return interval * i + start

class GenerateBipedal:
    def __init__(self, args):
        self.args = args
        self.output = {}
        # Red flag is at about x=50
        self.start_x = 50
        self.start_x += 0.5 * self.args['hull_width']
        # Ground starts at y=100
        self.start_y = 100
        # Currently, naively calculate total leg height
        for k in ['leg', 'lower']:
            self.start_y += self.args[k + '_height']
        self.hull_segment_width = self.args['hull_width'] / self.args['body_segments']
        self.odd_body_segments = self.args['body_segments'] % 2 == 1

    # Returns True if i is the "hull" segment and False otherwise
    def is_hull_segment(self, i):
        return (self.odd_body_segments and i == self.args['body_segments'] // 2) or (not self.odd_body_segments and i == self.args['body_segments'] // 2 - 1)

    # Returns 'left' if i is the segment/number left of hull and 'right' if i is directly right of hull and '' otherwise
    def is_adj_to_hull(self, i):
        if (self.odd_body_segments and i == (self.args['body_segments'] // 2) - 1) or (not self.odd_body_segments and i == self.args['body_segments'] // 2 - 2):
            return 'left'
        elif (self.odd_body_segments and i == self.args['body_segments'] // 2) or (not self.odd_body_segments and i == self.args['body_segments'] // 2 - 1):
            return 'right'
        else:
            return ''

    def build(self):
        self.build_fixtures()

        self.build_bodies()

        self.build_joints()

    def build_fixtures(self):
        for i in range(self.args['body_segments']):
            # TODO(josh): separate this logic out into a helper function, is_adj_to_hull or something
            if self.is_hull_segment(i):
                f = 'Hull' + 'Fixture'
            else:
                f = 'Body' + str(i) + 'Fixture'
            self.output[f] = {}
            self.output[f]['DataType'] = 'Fixture'
            self.output[f]['FixtureShape'] = {}
            self.output[f]['FixtureShape']['Type'] = 'PolygonShape'
            half_width = 0.5 * self.hull_segment_width
            bottom_half_height = 0.5 * self.args['hull_height']
            # If only one hull segment, build pentagon. Else, build segment
            if self.args['body_segments'] == 1:
                self.output[f]['FixtureShape']['Vertices'] = [
                    [-half_width, bottom_half_height],
                    [0, bottom_half_height],
                    [half_width, 0],
                    [half_width, -bottom_half_height],
                    [-half_width, -bottom_half_height]
                ]
            else:
                if (i < self.args['body_segments'] // 2):
                    front_half_height = bottom_half_height
                    back_half_height = bottom_half_height
                else:
                    front_half_height = ith_between(
                        self.args['hull_height'] // 2,
                        0,
                        i - (self.args['body_segments'] // 2) + 1,
                        self.args['body_segments']
                    )
                    back_half_height = ith_between(
                        self.args['hull_height'] // 2,
                        0,
                        i - (self.args['body_segments'] // 2),
                        self.args['body_segments']
                    )
                self.output[f]['FixtureShape']['Vertices'] = [
                    [-half_width, back_half_height],
                    [half_width, front_half_height],
                    [half_width, -bottom_half_height],
                    [-half_width, -bottom_half_height]
                ]
            self.output[f]['Friction'] = self.args['hull_friction']
            self.output[f]['Density'] = self.args['hull_density']
            self.output[f]['Restitution'] = 0.0
            self.output[f]['MaskBits'] = 1
            self.output[f]['CategoryBits'] = 32

        for prefix in ['Leg', 'Lower']:
            lower = prefix.lower()
            f = prefix + 'Fixture'
            self.output[f] = {}
            self.output[f]['DataType'] = 'Fixture'
            self.output[f]['FixtureShape'] = {}
            self.output[f]['FixtureShape']['Type'] = 'PolygonShape'
            half_width, half_height = 0.5 * self.args[lower + '_width'], 0.5 * self.args[lower + '_height']
            self.output[f]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]
            self.output[f]['Friction'] = self.args['hull_friction'] if prefix == 'Hull' else self.args['leg_friction']
            self.output[f]['Density'] = self.args['hull_density'] if prefix == 'Hull' else self.args['leg_density']
            self.output[f]['Restitution'] = 0.0
            self.output[f]['MaskBits'] = 1
            self.output[f]['CategoryBits'] = 32

    def build_bodies(self):
        if self.args['body_segments'] == 1:
            self.output['Hull'] = {}
            self.output['Hull']['DataType'] = 'DynamicBody'
            self.output['Hull']['Position'] = [self.start_x, self.start_y]
            self.output['Hull']['Angle'] = 0.0
            self.output['Hull']['FixtureNames'] = ['HullFixture']
            self.output['Hull']['Color1'] = HULL_COLOR
            self.output['Hull']['Color2'] = LINE_COLOR
            self.output['Hull']['CanTouchGround'] = False
            self.output['Hull']['InitialForceScale'] = 5
            self.output['Hull']['Depth'] = 0
        # Segment hull into even body pieces, back to front
        else:
            current_x = self.start_x - 0.5 * self.args['hull_width'] + 0.5 * self.hull_segment_width
            for i in range(self.args['body_segments']):
                # If even number of body pieces, 'Hull' will be left-of-center piece
                if self.is_hull_segment(i):
                    # Body number self.args['body_segments'] / 2 is renamed as 'Hull'
                    k = 'Hull'
                else:
                    k = 'Body' + str(i)
                self.output[k] = {}
                self.output[k]['DataType'] = 'DynamicBody'
                self.output[k]['Position'] = [current_x, self.start_y]
                self.output[k]['Angle'] = 0.0
                self.output[k]['FixtureNames'] = [k + 'Fixture']
                self.output[k]['Color1'] = HULL_COLOR
                self.output[k]['Color2'] = LINE_COLOR
                self.output[k]['CanTouchGround'] = False
                self.output[k]['InitialForceScale'] = 5
                self.output[k]['Depth'] = 0
                current_x += self.hull_segment_width

        for sign in [-1, +1]:
            current_x = self.start_x
            current_y = self.start_y - 0.5 * self.args['leg_height']
            for prefix in ['Leg', 'Lower']:
                k = prefix + str(sign)
                self.output[k] = {}
                self.output[k]['DataType'] = 'DynamicBody'
                self.output[k]['Position'] = [current_x, current_y]
                self.output[k]['Angle'] = sign * 0.05
                self.output[k]['FixtureNames'] = [prefix + 'Fixture']
                self.output[k]['Color1'] = LIGHT_COLOR if sign == 1 else DARK_COLOR
                self.output[k]['Color2'] = LINE_COLOR
                self.output[k]['CanTouchGround'] = True
                self.output[k]['Depth'] = sign + 1
                current_y = current_y - 0.5 * self.args['lower_height']

    def build_joints(self):
        # current_x is unused, used to be used for calculating 'Anchor' for weld joints
        current_x = self.start_x - 0.5 * self.args['hull_width'] + 0.5 * self.hull_segment_width
        for i in range(self.args['body_segments'] - 1):
            if self.is_adj_to_hull(i) == 'left':
                body_a = 'Body' + str(i)
                body_b = 'Hull'
            elif self.is_adj_to_hull(i) == 'right':
                body_a = 'Hull'
                body_b = 'Body' + str(i+1)
            else:
                body_a = 'Body' + str(i)
                body_b = 'Body' + str(i+1)
            k = body_a + body_b + 'Joint'
            self.output[k] = {}
            self.output[k]['DataType'] = 'Linkage' if self.args['rigid_spine'] else 'JointMotor'
            self.output[k]['BodyA'] = body_a
            self.output[k]['BodyB'] = body_b
            self.output[k]['LocalAnchorA'] = [0.5 * self.hull_segment_width, 0]
            self.output[k]['LocalAnchorB'] = [-0.5 * self.hull_segment_width, 0]

            if not self.args['rigid_spine']:
                self.output[k]['EnableMotor'] = self.args['spine_motors']
                self.output[k]['EnableLimit'] = True
                self.output[k]['MaxMotorTorque'] = 80
                self.output[k]['MotorSpeed'] = 0.0
                self.output[k]['LowerAngle'] = -0.5
                self.output[k]['UpperAngle'] = 0.2
                self.output[k]['Speed'] = 1
                self.output[k]['Depth'] = 0

            current_x += self.hull_segment_width


        for sign in [-1, +1]:
            k = 'HullLeg' + str(sign) + 'Joint'
            self.output[k] = {}
            self.output[k]['DataType'] = 'JointMotor'
            self.output[k]['BodyA'] = 'Hull'
            self.output[k]['BodyB'] = 'Leg' + str(sign)
            self.output[k]['LocalAnchorA'] = [0.5 * self.hull_segment_width if self.args['body_segments'] % 2 == 0 else 0, -0.5 * self.args['hull_height']]
            self.output[k]['LocalAnchorB'] = [0, 0.5 * self.args['leg_height']]
            self.output[k]['EnableMotor'] = True
            self.output[k]['EnableLimit'] = True
            self.output[k]['MaxMotorTorque'] = 80
            self.output[k]['MotorSpeed'] = 1
            self.output[k]['LowerAngle'] = -0.8
            self.output[k]['UpperAngle'] = 1.1
            self.output[k]['Speed'] = 4
            self.output[k]['Depth'] = sign + 1

            k = 'Leg' + str(sign) + 'Lower' + str(sign) + 'Joint'
            self.output[k] = {}
            self.output[k]['DataType'] = 'JointMotor'
            self.output[k]['BodyA'] = 'Leg' + str(sign)
            self.output[k]['BodyB'] = 'Lower' + str(sign)
            self.output[k]['LocalAnchorA'] = [0, -0.5 * self.args['leg_height']]
            self.output[k]['LocalAnchorB'] = [0, 0.5 * self.args['lower_height']]
            self.output[k]['EnableMotor'] = True
            self.output[k]['EnableLimit'] = True
            self.output[k]['MaxMotorTorque'] = 80
            self.output[k]['MotorSpeed'] = 1
            self.output[k]['LowerAngle'] = -1.6
            self.output[k]['UpperAngle'] = -0.1
            self.output[k]['Speed'] = 6
            self.output[k]['Depth'] = sign + 1

    def write_to_json(self, filename=None):
        if not os.path.exists('box2d-json-gen'):
            os.mkdir('box2d-json-gen')

        if filename is not None:
            outfile = open(filename, 'w+')
        else:
            outfile = open(self.args['filename'], 'w+')
        outfile.write(json.dumps(self.output, indent=4, separators=(',', ' : ')))

if __name__ == '__main__':
    args = parse_args()

    gen = GenerateBipedal(args)

    gen.build()

    gen.write_to_json()

