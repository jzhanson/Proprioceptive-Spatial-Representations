# TODO(josh): can simplify script by not making distinction between 'Head', 'Neck', 'Tail', and
# 'Hull', make them all 'Body' but name the center body 'Hull' for json_walker.py purposes
# TODO(josh): Make this a class
import sys
import os
import math
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--neck-segments',
        type=int,
        default=4,
        help='Number of neck segments')
    parser.add_argument(
        '--tail-segments',
        type=int,
        default=4,
        help='Number of tail segments')
    parser.add_argument(
        '--filename',
        type=str,
        default='box2d-json/GeneratedRaptorWalker.json',
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
        help='The height of an edge of the center hull segment (default 28.0)')
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
        '--lower-width',
        type=float,
        default=7.0,
        help='Lower (second highest leg segment) width (default 7)')
    parser.add_argument(
        '--lower-height',
        type=float,
        default=24.0,
        help='Lower (second highest leg segment) height (default 24.0)')
    parser.add_argument(
        '--shin-width',
        type=float,
        default=6.4,
        help='Shin (third highest leg segment) width (default 6.4)')
    parser.add_argument(
        '--shin-height',
        type=float,
        default=20.0,
        help='Shin (third highest leg segment) height (default 20.0)')
    parser.add_argument(
        '--foot-width',
        type=float,
        default=5.6,
        help='Foot (bottom leg segment) width (default 5.6)')
    parser.add_argument(
        '--foot-height',
        type=float,
        default=16.0,
        help='Foot (bottom leg segment) height (default 16.0)')
    # Can change start x/y args to be center x/y
    parser.add_argument(
        '--start-x',
        type=int,
        default=140,
        help='The x-coordinate to generate the hull (center body segment) (default 140)')
    parser.add_argument(
        '--start-y',
        type=int,
        default=168,
        help='The y-coordinate to generate the hull (hindmost body segment) (default 168)')
    return vars(parser.parse_args())

LIGHT_COLOR = [0.6392156862745098, 0.6941176470588235, 0.7372549019607844]
MID_COLOR = [0.47058823529411764, 0.5490196078431373, 0.6862745098039216]
DARK_COLOR = [0.35294117647058826, 0.4117647058823529, 0.47058823529411764]
LINE_COLOR = [0.23921568627450981, 0.2823529411764706, 0.35294117647058826]

# TODO(josh): make ith_between consistent with neck/tail segment numberings
def ith_between(start, end, i, total):
    interval = (end - start) / total
    return interval * i + start

def build_fixtures(args, output):
    output['HullFixture'] = {}
    output['HeadFixture'] = {}

    leg_fixtures = ['ThighFixture', 'LowerFixture', 'ShinFixture', 'FootFixture']

    for f in leg_fixtures:
        output[f] = {}

    # Build neck and tail separately
    start_width, start_height = args['hull_width'], args['hull_height']
    end_width, end_height = args['neck_width'], args['neck_height']

    for i in range(args['neck_segments']):
        k = 'Neck' + str(i) + 'Fixture'
        output[k] = {}
        output[k]['DataType'] = 'Fixture'
        output[k]['FixtureShape'] = {}
        output[k]['FixtureShape']['Type'] = 'PolygonShape'
        half_width = 0.5 * ith_between(
            start_width,
            end_width,
            i+1,
            args['neck_segments']
        )
        half_height = 0.5 * ith_between(
            start_height,
            end_height,
            i+1,
            args['neck_segments']
        )
        output[k]['FixtureShape']['Vertices'] = [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ]

    start_width, start_height = args['hull_width'], args['hull_height']
    end_width, end_height = args['tail_width'], args['tail_height']

    for i in range(args['tail_segments']):
        k = 'Tail' + str(i) + 'Fixture'
        output[k] = {}
        output[k]['DataType'] = 'Fixture'
        output[k]['FixtureShape'] = {}
        output[k]['FixtureShape']['Type'] = 'PolygonShape'
        half_width = 0.5 * ith_between(
            start_width,
            end_width,
            i+1,
            args['tail_segments']
        )
        half_height = 0.5 * ith_between(
            start_height,
            end_height,
            i+1,
            args['tail_segments']
        )
        output[k]['FixtureShape']['Vertices'] = [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ]

    for f in output.keys():
        if not 'Neck' in f and not 'Tail' in f:
            output[f]['DataType'] = 'Fixture'
            output[f]['FixtureShape'] = {}
            output[f]['FixtureShape']['Type'] = 'PolygonShape'

            prefix = f.split('Fixture')[0].lower()

            half_width, half_height = args[prefix + '_width'] / 2, args[prefix + '_height'] / 2

            output[f]['FixtureShape']['Vertices'] = [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height]
            ]

        output[f]['Friction'] = 0.1 if f == 'HullFixture' else (0.2 if f in leg_fixtures else 0.0)
        output[f]['Density'] = 1.0 if f in leg_fixtures else 5.0
        output[f]['Restitution'] = 0.0
        output[f]['MaskBits'] = 1
        output[f]['CategoryBits'] = 32

def body_add_position_angle_neck_or_tail(args, output, neck_or_tail):
    title_neck_or_tail = neck_or_tail.title()
    # Whether to build in the positive direction (neck) or negative (tail)
    x_dir = 1.0 if neck_or_tail == 'neck' else -1.0
    # Fill in position + angle for neck and head
    if args[neck_or_tail + '_segments'] > 0:
        current_x = args['start_x'] + x_dir * (0.5 * args['hull_width'] + 0.5 * ith_between(
            args['hull_width'],
            args[neck_or_tail + '_width'],
            1,
            args[neck_or_tail + '_segments']
        ))
        # Align segments along top spine
        current_y = args['start_y'] + 0.5 * (args['hull_height'] - ith_between(
            args['hull_height'],
            args[neck_or_tail + '_height'],
            1,
            args[neck_or_tail + '_segments']
        ))
    elif neck_or_tail == 'neck':
        head_x = args['start_x'] + 0.5 * args['hull_width'] + 0.5 * args['head_width']
        head_y = args['start_y'] + 0.5 * (args['hull_height'] - args['head_height'])
        output['Head']['Angle'] = 0.0
        output['Head']['Position'] = [head_x, head_y]
        return
    else:
        return

    for i in range(args[neck_or_tail + '_segments']):
        output[title_neck_or_tail + str(i)]['Angle'] = 0
        #ith_between(0.25, 0.65, i, args[neck_or_tail + '_segments'])
        # TODO(josh): make this support a different position based on angle
        output[title_neck_or_tail + str(i)]['Position'] = [current_x, current_y]

        current_x = current_x + x_dir * (0.5 * ith_between(
            args['hull_width'],
            args[neck_or_tail + '_width'],
            i+1,
            args[neck_or_tail + '_segments']
        ) + 0.5 * ith_between(
            args['hull_width'],
            args[neck_or_tail + '_width'],
            i+2,
            args[neck_or_tail + '_segments']
        ))
        current_y = current_y + 0.5 * (ith_between(
            args['hull_height'],
            args[neck_or_tail + '_height'],
            i+1,
            args[neck_or_tail + '_segments']
        ) - ith_between(
            args['hull_height'],
            args[neck_or_tail + '_height'],
            i+2,
            args[neck_or_tail + '_segments']
        ))

        # Add the head if building neck
        if neck_or_tail == 'neck' and i == args[neck_or_tail + '_segments'] - 1:
            # Set up current_x and current_y for building head if last one
            current_x = current_x + ith_between(
                args['hull_width'],
                args['neck_width'],
                i+1,
                args['neck_segments']
            ) + 0.5 * args['head_width']
            current_y = current_y + (ith_between(
                args['hull_height'],
                args['neck_height'],
                i+1,
                args['neck_segments']
            ) - 0.5 * args['head_height'])

            output['Head']['Angle'] = args['head_angle']
            output['Head']['Position'] = [current_x, current_y]

def build_leg_bodies(args, output):
    for f in ['ThighFixture', 'LowerFixture', 'ShinFixture', 'FootFixture']:
        for sign in [-1, +1]:
            k = f.split('Fixture')[0] + str(sign)
            output[k] = {}
            output[k]['DataType'] = 'DynamicBody'
            output[k]['FixtureNames'] = [f]
            output[k]['Color1'] = DARK_COLOR if '-1' in k else MID_COLOR
            output[k]['Color2'] = LINE_COLOR
            output[k]['CanTouchGround'] = True
            output[k]['InitialForceScale'] =  0
            output[k]['Depth'] = 1 if '-1' in k else 0

            # Add position and angle
            # TODO(josh): make angle of legs an argument?
            thigh_x = args['start_x'] + math.sin(0.3) * args['thigh_height'] / 2
            thigh_y = args['start_y'] - math.cos(0.3) * args['thigh_height'] / 2
            lower_x = thigh_x - args['lower_height'] / 2
            lower_y = thigh_y - args['thigh_height'] / 2
            shin_x = lower_x - args['lower_height'] / 2
            shin_y = lower_y - args['shin_height'] / 2
            foot_x = shin_x + args['foot_height'] / 2
            foot_y = shin_y - args['shin_height'] / 2
            if 'Thigh' in k:
                output[k]['Position'] = [thigh_x, thigh_y]
                output[k]['Angle'] = 0.3
            elif 'Lower' in k:
                # Note: lower is sideways
                output[k]['Position'] = [lower_x, lower_y]
                output[k]['Angle'] = -1.2
            elif 'Shin' in k:
                output[k]['Position'] = [shin_x, shin_y]
                output[k]['Angle'] = 0.25
            elif 'Foot' in k:
                # Foot is also sideways
                output[k]['Position'] = [foot_x, foot_y]
                output[k]['Angle'] = 1.45

def build_bodies(args, output):
    start_x = args['start_x']
    start_y = args['start_y']

    # Build common features (but build legs separately)
    fixtures = list(output.keys())
    for k in fixtures:
        body_name = k.split('Fixture')[0]

        if body_name in ['Thigh', 'Lower', 'Shin', 'Foot']:
            continue

        output[body_name] = {}
        output[body_name]['DataType'] = 'DynamicBody'
        output[body_name]['FixtureNames'] = [k]
        output[body_name]['Color1'] = MID_COLOR
        output[body_name]['Color2'] = LINE_COLOR
        output[body_name]['CanTouchGround'] = 'Tail' in body_name
        output[body_name]['InitialForceScale'] = 100 if body_name == 'Hull' else 0
        output[body_name]['Depth'] = 1

    # Fill in position + angle for hull
    output['Hull']['Position'] = [start_x, start_y]
    output['Hull']['Angle'] = 0

    # Fill in position + angle neck and tail
    body_add_position_angle_neck_or_tail(args, output, 'neck')
    body_add_position_angle_neck_or_tail(args, output, 'tail')

    # Build bodies for 2 legs
    build_leg_bodies(args, output)

def build_neck_or_tail_joints(args, output, neck_or_tail):
    title_neck_or_tail = neck_or_tail.title()
    x_dir = 1.0 if neck_or_tail == 'neck' else -1.0
    joint_counter = 0 if neck_or_tail == 'neck' else args['neck_segments'] + 1

    if neck_or_tail == 'neck' and args['neck_segments'] == 0:
        k = 'Joint' + str(joint_counter) + '.Hull.Head'
        output[k] = {}
        output[k]['BodyA'] = 'Hull'
        output[k]['BodyB'] = 'Head'
        output[k]['LocalAnchorA'] = [0.5 * args['hull_width'], 0]
        output[k]['LocalAnchorB'] = [
            -0.5 * args['head_width'],
            0.25 * args['head_height']
        ]
        # Use neck upper/lower angles for this edge case
        output[k]['LowerAngle'] = -0.5
        output[k]['UpperAngle'] = 0.2
        return
    elif args[neck_or_tail + '_segments'] == 0:
        return

    k = 'Joint' + str(joint_counter) + '.Hull.' + title_neck_or_tail + '0'
    output[k] = {}
    output[k]['BodyA'] = 'Hull'
    output[k]['BodyB'] = title_neck_or_tail + '0'
    current_width = ith_between(
        args['hull_width'],
        args[neck_or_tail + '_width'],
        1,
        args[neck_or_tail + '_segments']
    )
    current_height = ith_between(
        args['hull_height'],
        args[neck_or_tail + '_height'],
        1,
        args[neck_or_tail + '_segments']
    )
    output[k]['LocalAnchorA'] = [
        x_dir * 0.5 * args['hull_width'],
        0.5 * (args['hull_height'] - current_height)
    ]
    output[k]['LocalAnchorB'] = [-x_dir * 0.5 * current_width, 0]
    output[k]['LowerAngle'] = -0.5
    output[k]['UpperAngle'] = 0.2
    joint_counter += 1

    for i in range(args[neck_or_tail + '_segments'] - 1):
        k = 'Joint' + str(joint_counter) + '.' + title_neck_or_tail + str(i) + '.' + title_neck_or_tail + str(i+1)
        prev_width = current_width
        prev_height = current_height
        current_width= ith_between(
            args['hull_width'],
            args[neck_or_tail + '_width'],
            i+2,
            args[neck_or_tail + '_segments']
        )
        current_height = ith_between(
            args['hull_height'],
            args[neck_or_tail + '_height'],
            i+2,
            args[neck_or_tail + '_segments']
        )
        output[k] = {}
        output[k]['BodyA'] = title_neck_or_tail + str(i)
        output[k]['BodyB'] = title_neck_or_tail + str(i+1)
        output[k]['LocalAnchorA'] = [
            x_dir * 0.5 * prev_width,
            0.5 * (prev_height - current_height)
        ]
        output[k]['LocalAnchorB'] = [-x_dir * 0.5 * current_width, 0]
        output[k]['LowerAngle'] = -0.5
        output[k]['UpperAngle'] = 0.2
        joint_counter += 1

    # If building neck joints, build head joint
    if neck_or_tail == 'neck':
        # Head-neck joint
        k = 'Joint' + str(joint_counter) + '.Neck' + str(args['neck_segments'] - 1) + '.Head'
        output[k] = {}
        output[k]['BodyA'] = 'Neck' + str(args['neck_segments'] - 1)
        output[k]['BodyB'] = 'Head'
        current_width = ith_between(
            args['hull_width'],
            args['neck_width'],
            args['neck_segments']+1,
            args['neck_segments']
        )
        output[k]['LocalAnchorA'] = [x_dir * 0.5 * current_width, 0]
        output[k]['LocalAnchorB'] = [
            -0.5 * args['head_width'],
            0.25 * args['head_height']
        ]
        output[k]['LowerAngle'] = -0.9
        output[k]['UpperAngle'] = 0.7
        joint_counter += 1

def build_leg_joints(args, output):
    joint_counter = args['neck_segments'] + args['tail_segments'] + 1
    leg_names = ['Thigh', 'Lower', 'Shin', 'Foot']

    for sign in [-1, +1]:
        k = 'Joint' + str(joint_counter) + 'Hull' + 'Thigh' + str(sign)
        output[k] = {}
        output[k]['BodyA'] = 'Hull'
        output[k]['BodyB'] = 'Thigh' + str(sign)
        output[k]['LocalAnchorA'] = [0.0, 0.0]
        output[k]['LocalAnchorB'] = [0.0, args['thigh_height'] / 2]
        output[k]['LowerAngle'] = -0.8
        output[k]['UpperAngle'] = 1.1

        joint_counter += 1

        for i in range(len(leg_names)-1):
            k = 'Joint' + str(joint_counter) + '.' + leg_names[i] + str(sign) + '.' + leg_names[i+1] + str(sign)
            output[k] = {}
            output[k]['BodyA'] = leg_names[i] + str(sign)
            output[k]['BodyB'] = leg_names[i+1] + str(sign)
            output[k]['LocalAnchorA'] = [0.0, -args[leg_names[i].lower() + '_height'] / 2]
            output[k]['LocalAnchorB'] = [0.0, args[leg_names[i+1].lower() + '_height'] / 2]
            output[k]['LowerAngle'] = -0.8
            if leg_names[i] == 'Thigh':
                output[k]['UpperAngle'] = 1.1
            elif leg_names[i] == 'Lower':
                output[k]['UpperAngle'] = 0.5
            elif leg_names[i] == 'Shin':
                output[k]['UpperAngle'] = 0.8

            output[k]['Depth'] = 0 if sign == -1 else 1

            joint_counter += 1

def build_joints(args, output):
    # Build neck and tail joints
    build_neck_or_tail_joints(args, output, 'neck')
    build_neck_or_tail_joints(args, output, 'tail')

    # Build leg joints
    build_leg_joints(args, output)

    for k in output.keys():
        if 'Joint' in k:
            output[k]['DataType'] = 'JointMotor'
            output[k]['EnableMotor'] = True
            output[k]['EnableLimit'] = True
            output[k]['MaxMotorTorque'] = 80
            output[k]['MotorSpeed'] = 0.0
            output[k]['Speed'] = 1
            output[k]['Depth'] = 0 if '-1' in k else 1


if __name__ == '__main__':
    args = parse_args()

    output = {}

    build_fixtures(args, output)

    build_bodies(args, output)

    build_joints(args, output)

    print(json.dumps(output, indent=4, separators=(',', ': ')))

    outfile = open(args['filename'], 'w+')
    outfile.write(json.dumps(output, indent=4, separators=(',', ' : ')))

