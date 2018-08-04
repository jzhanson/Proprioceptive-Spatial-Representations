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
        default='box2d-json/GeneratedCentipedeWalker.json',
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
    parser.add_argument(
        '--start-x',
        type=int,
        default=140,
        help='The x-coordinate to generate the hindmost body segment (default 140)')
    parser.add_argument(
        '--start-y',
        type=int,
        default=135,
        help='The y-coordinate to generate the hindmost body segment (default 135)')
    return vars(parser.parse_args())

def build_fixtures(args, output):
    output['HullFixture'] = {}
    output['LegFixture'] = {}
    output['LowerFixture'] = {}
    for f in output.keys():
        output[f]['DataType'] = 'Fixture'
        output[f]['FixtureShape'] = {}
        if f == 'HullFixture':
            output[f]['FixtureShape']['Type'] = 'CircleShape'
            output[f]['FixtureShape']['Radius'] = args['hull_radius']
        elif f == 'LegFixture':
            output[f]['FixtureShape']['Type'] = 'PolygonShape'
            half_width, half_height = args['leg_width'] / 2, args['leg_height'] / 2
            output[f]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]
        elif f == 'LowerFixture':
            output[f]['FixtureShape']['Type'] = 'PolygonShape'
            half_width, half_height = args['lower_width'] / 2, args['lower_height'] / 2
            output[f]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]
        output[f]['Friction'] = 0.2
        output[f]['Density'] = 5.0 if f == 'HullFixture' else 1.0
        output[f]['Restitution'] = 0.0
        output[f]['MaskBits'] = 1
        output[f]['CategoryBits'] = 32

def add_segments(num_segments, args, output):
    start_x = args['start_x']
    start_y = args['start_y']
    hull_radius = args['hull_radius']
    # Build body back to front
    for i in range(num_segments):
        hull_str = 'Hull' + str(i) if i > 0 else 'Hull'
        output[hull_str] = {}
        output[hull_str]['DataType'] = 'DynamicBody'
        output[hull_str]['Position'] = [start_x + i * (2 * hull_radius), start_y]
        output[hull_str]['Angle'] = 0.0
        output[hull_str]['FixtureNames'] = ['HullFixture']
        output[hull_str]['Color1'] = [0.5, 0.4, 0.9]
        output[hull_str]['Color2'] = [0.3, 0.3, 0.5]
        output[hull_str]['CanTouchGround'] = False
        output[hull_str]['InitialForceScale'] = 5
        output[hull_str]['Depth'] = 0

        # Build legs
        leg_height = args['leg_height']
        lower_height = args['lower_height']
        for sign in [-1, +1]:
            leg_str = 'Leg-' + str(i) if sign == -1 else 'Leg' + str(i)
            output[leg_str] = {}
            output[leg_str]['DataType'] = 'DynamicBody'
            output[leg_str]['Position'] = [start_x + i * (2 * hull_radius), start_y + leg_height / 2]
            output[leg_str]['Angle'] = sign * 0.05
            output[leg_str]['FixtureNames'] = ['LegFixture']
            output[leg_str]['Color1'] = [0.7, 0.4, 0.6] if sign == -1 else [0.4, 0.2, 0.4]
            output[leg_str]['Color2'] = [0.5, 0.3, 0.5] if sign == -1 else [0.3, 0.1, 0.2]
            output[leg_str]['CanTouchGround'] = True
            output[leg_str]['Depth'] = sign + 1

            hull_leg_joint_str = hull_str + leg_str + 'Joint'
            output[hull_leg_joint_str] = {}
            output[hull_leg_joint_str]['DataType'] = 'JointMotor'
            output[hull_leg_joint_str]['BodyA'] = hull_str
            output[hull_leg_joint_str]['BodyB'] = leg_str
            output[hull_leg_joint_str]['LocalAnchorA'] = [0, 0]
            output[hull_leg_joint_str]['LocalAnchorB'] = [0, -1 * leg_height / 2]
            output[hull_leg_joint_str]['EnableMotor'] = True
            output[hull_leg_joint_str]['EnableLimit'] = True
            output[hull_leg_joint_str]['MaxMotorTorque'] = 80
            output[hull_leg_joint_str]['MotorSpeed'] = 1
            output[hull_leg_joint_str]['LowerAngle'] = -3.14
            output[hull_leg_joint_str]['UpperAngle'] = 3.14
            output[hull_leg_joint_str]['Speed'] = 4
            output[hull_leg_joint_str]['Depth'] = sign + 1

            lower_str = 'Lower-' + str(i) if sign == -1 else 'Lower' + str(i)
            output[lower_str] = {}
            output[lower_str]['DataType'] = 'DynamicBody'
            output[lower_str]['Position'] = [start_x + i * (2 * hull_radius), start_y + leg_height - lower_height / 2]
            output[lower_str]['Angle'] = sign * 0.05
            output[lower_str]['FixtureNames'] = ['LowerFixture']
            output[lower_str]['Color1'] = [0.7, 0.4, 0.6] if sign == -1 else [0.4, 0.2, 0.4]
            output[lower_str]['Color2'] = [0.5, 0.3, 0.5] if sign == -1 else [0.3, 0.1, 0.2]
            output[lower_str]['CanTouchGround'] = True
            output[lower_str]['Depth'] = sign + 1

            leg_lower_joint_str = leg_str + lower_str + 'Joint'
            output[leg_lower_joint_str] = {}
            output[leg_lower_joint_str]['DataType'] = 'JointMotor'
            output[leg_lower_joint_str]['BodyA'] = leg_str
            output[leg_lower_joint_str]['BodyB'] = lower_str
            output[leg_lower_joint_str]['LocalAnchorA'] = [0, leg_height / 2]
            output[leg_lower_joint_str]['LocalAnchorB'] = [0, lower_height / 2]
            output[leg_lower_joint_str]['EnableMotor'] = True
            output[leg_lower_joint_str]['EnableLimit'] = True
            output[leg_lower_joint_str]['MaxMotorTorque'] = 80
            output[leg_lower_joint_str]['MotorSpeed'] = 1
            output[leg_lower_joint_str]['LowerAngle'] = -3.14
            output[leg_lower_joint_str]['UpperAngle'] = 3.14
            output[leg_lower_joint_str]['Speed'] = 6
            output[leg_lower_joint_str]['Depth'] = sign + 1

        # Build weld joint for all but frontmost body
        if i < num_segments - 1:
            next_hull_str = 'Hull' + str(i+1)
            weld_str = hull_str + next_hull_str + 'Joint'
            output[weld_str] = {}
            output[weld_str]['DataType'] = 'Linkage'
            output[weld_str]['BodyA'] = hull_str
            output[weld_str]['BodyB'] = next_hull_str
            output[weld_str]['Anchor'] = [start_x + i * (2 * hull_radius) + hull_radius, start_y]
            output[weld_str]['Depth'] = 0


if __name__ == '__main__':
    args = parse_args()

    output = {}

    build_fixtures(args, output)

    add_segments(args['num_segments'], args, output)

    print(json.dumps(output, indent=4, separators=(',', ': ')))

    outfile = open(args['filename'], 'w+')
    outfile.write(json.dumps(output, indent=4, separators=(',', ' : ')))

