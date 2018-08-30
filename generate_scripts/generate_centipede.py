import sys
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-segments',
        type=int,
        default=2,
        help='Number of centipede segments')
    parser.add_argument(
        '--filename',
        type=str,
        default='box2d-json-gen/GeneratedCentipedeWalker.json',
        help='What to call the output JSON file')
    parser.add_argument(
        '--hull-radius',
        type=float,
        default=12.0,
        help='The radius of a hull segment (default 12.0)')
    # Later on, can support PolygonShapes with an edge length argument for hull
    parser.add_argument(
        '--leg-width',
        type=float,
        default=6.0,
        help='The width of upper leg segments  (default 6.0)')
    parser.add_argument(
        '--leg-height',
        type=float,
        default=26.0,
        help='The height of upper leg segments  (default 26.0)')
    parser.add_argument(
        '--lower-width',
        type=float,
        default=6.4,
        help='The width of upper leg segments  (default 6.4)')
    parser.add_argument(
        '--lower-height',
        type=float,
        default=60.0,
        help='The width of upper leg segments  (default 60.0)')
    # Can change start x/y args to be center x/y
    return vars(parser.parse_args())

class GenerateCentipede:
    def __init__(self, args):
        self.args = args
        self.output = {}
        # start_x is where to start the last (hindmost) body segment
        self.start_x = 50
        self.start_y = 100
        self.start_y += self.args['lower_height'] - self.args['leg_height']

    def build(self):
        self.build_fixtures()
        self.add_segments()

    def build_fixtures(self):
        self.output['HullFixture'] = {}
        self.output['LegFixture'] = {}
        self.output['LowerFixture'] = {}
        for f in self.output.keys():
            self.output[f]['DataType'] = 'Fixture'
            self.output[f]['FixtureShape'] = {}
            if f == 'HullFixture':
                self.output[f]['FixtureShape']['Type'] = 'CircleShape'
                self.output[f]['FixtureShape']['Radius'] = self.args['hull_radius']
            elif f == 'LegFixture':
                self.output[f]['FixtureShape']['Type'] = 'PolygonShape'
                half_width, half_height = self.args['leg_width'] / 2, self.args['leg_height'] / 2
                self.output[f]['FixtureShape']['Vertices'] = [
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height]
                ]
            elif f == 'LowerFixture':
                self.output[f]['FixtureShape']['Type'] = 'PolygonShape'
                half_width, half_height = self.args['lower_width'] / 2, self.args['lower_height'] / 2
                self.output[f]['FixtureShape']['Vertices'] = [
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height]
                ]
            self.output[f]['Friction'] = 0.2
            self.output[f]['Density'] = 5.0 if f == 'HullFixture' else 1.0
            self.output[f]['Restitution'] = 0.0
            self.output[f]['MaskBits'] = 1
            self.output[f]['CategoryBits'] = 32

    def add_segments(self):
        start_x = self.start_x
        start_y = self.start_y
        hull_radius = self.args['hull_radius']
        # Build body back to front
        for i in range(self.args['num_segments']):
            hull_str = 'Hull' + str(i) if i > 0 else 'Hull'
            self.output[hull_str] = {}
            self.output[hull_str]['DataType'] = 'DynamicBody'
            self.output[hull_str]['Position'] = [start_x + i * (2 * hull_radius), start_y]
            self.output[hull_str]['Angle'] = 0.0
            self.output[hull_str]['FixtureNames'] = ['HullFixture']
            self.output[hull_str]['Color1'] = [0.5, 0.4, 0.9]
            self.output[hull_str]['Color2'] = [0.3, 0.3, 0.5]
            self.output[hull_str]['CanTouchGround'] = False
            self.output[hull_str]['InitialForceScale'] = 5
            self.output[hull_str]['Depth'] = 0

            # Build legs
            leg_height = self.args['leg_height']
            lower_height = self.args['lower_height']
            for sign in [-1, +1]:
                leg_str = 'Leg-' + str(i) if sign == -1 else 'Leg' + str(i)
                self.output[leg_str] = {}
                self.output[leg_str]['DataType'] = 'DynamicBody'
                self.output[leg_str]['Position'] = [start_x + i * (2 * hull_radius), start_y + leg_height / 2]
                self.output[leg_str]['Angle'] = sign * 0.05
                self.output[leg_str]['FixtureNames'] = ['LegFixture']
                self.output[leg_str]['Color1'] = [0.7, 0.4, 0.6] if sign == -1 else [0.4, 0.2, 0.4]
                self.output[leg_str]['Color2'] = [0.5, 0.3, 0.5] if sign == -1 else [0.3, 0.1, 0.2]
                self.output[leg_str]['CanTouchGround'] = True
                self.output[leg_str]['Depth'] = sign + 1

                hull_leg_joint_str = hull_str + leg_str + 'Joint'
                self.output[hull_leg_joint_str] = {}
                self.output[hull_leg_joint_str]['DataType'] = 'JointMotor'
                self.output[hull_leg_joint_str]['BodyA'] = hull_str
                self.output[hull_leg_joint_str]['BodyB'] = leg_str
                self.output[hull_leg_joint_str]['LocalAnchorA'] = [0, 0]
                self.output[hull_leg_joint_str]['LocalAnchorB'] = [0, -1 * leg_height / 2]
                self.output[hull_leg_joint_str]['EnableMotor'] = True
                self.output[hull_leg_joint_str]['EnableLimit'] = True
                self.output[hull_leg_joint_str]['MaxMotorTorque'] = 80
                self.output[hull_leg_joint_str]['MotorSpeed'] = 1
                self.output[hull_leg_joint_str]['LowerAngle'] = -3.14
                self.output[hull_leg_joint_str]['UpperAngle'] = 3.14
                self.output[hull_leg_joint_str]['Speed'] = 4
                self.output[hull_leg_joint_str]['Depth'] = sign + 1

                lower_str = 'Lower-' + str(i) if sign == -1 else 'Lower' + str(i)
                self.output[lower_str] = {}
                self.output[lower_str]['DataType'] = 'DynamicBody'
                self.output[lower_str]['Position'] = [start_x + i * (2 * hull_radius), start_y + leg_height - lower_height / 2]
                self.output[lower_str]['Angle'] = sign * 0.05
                self.output[lower_str]['FixtureNames'] = ['LowerFixture']
                self.output[lower_str]['Color1'] = [0.7, 0.4, 0.6] if sign == -1 else [0.4, 0.2, 0.4]
                self.output[lower_str]['Color2'] = [0.5, 0.3, 0.5] if sign == -1 else [0.3, 0.1, 0.2]
                self.output[lower_str]['CanTouchGround'] = True
                self.output[lower_str]['Depth'] = sign + 1

                leg_lower_joint_str = leg_str + lower_str + 'Joint'
                self.output[leg_lower_joint_str] = {}
                self.output[leg_lower_joint_str]['DataType'] = 'JointMotor'
                self.output[leg_lower_joint_str]['BodyA'] = leg_str
                self.output[leg_lower_joint_str]['BodyB'] = lower_str
                self.output[leg_lower_joint_str]['LocalAnchorA'] = [0, leg_height / 2]
                self.output[leg_lower_joint_str]['LocalAnchorB'] = [0, lower_height / 2]
                self.output[leg_lower_joint_str]['EnableMotor'] = True
                self.output[leg_lower_joint_str]['EnableLimit'] = True
                self.output[leg_lower_joint_str]['MaxMotorTorque'] = 80
                self.output[leg_lower_joint_str]['MotorSpeed'] = 1
                self.output[leg_lower_joint_str]['LowerAngle'] = -3.14
                self.output[leg_lower_joint_str]['UpperAngle'] = 3.14
                self.output[leg_lower_joint_str]['Speed'] = 6
                self.output[leg_lower_joint_str]['Depth'] = sign + 1

            # Build weld joint for all but frontmost body
            if i < self.args['num_segments'] - 1:
                next_hull_str = 'Hull' + str(i+1)
                weld_str = hull_str + next_hull_str + 'Joint'
                self.output[weld_str] = {}
                self.output[weld_str]['DataType'] = 'Linkage'
                self.output[weld_str]['BodyA'] = hull_str
                self.output[weld_str]['BodyB'] = next_hull_str
                self.output[weld_str]['Anchor'] = [start_x + i * (2 * hull_radius) + hull_radius, start_y]
                self.output[weld_str]['Depth'] = 0

    def write_to_json(self, filename=None):
        if not os.path.exists('box2d-json-gen'):
            os.mkdir('box2d-json-gen')
        if filename is None:
            outfile = open('box2d-json-gen/' + self.args['filename'], 'w+')
        else:
            outfile = open('box2d-json-gen/' + filename, 'w+')

        outfile.write(json.dumps(self.output, indent=4, separators=(',', ' : ')))

if __name__ == '__main__':
    args = parse_args()

    gen = GenerateCentipede(args)

    gen.build()

    gen.write_to_json()

