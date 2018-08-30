from __future__ import print_function, division
import os
import copy
import json
import argparse
import importlib

def parse_cmdline_args(additional_parser_args={}):
    parser = argparse.ArgumentParser(description='Raptor')
    parser.add_argument(
        '--json_file',
        type=str,
        default='',
        metavar='JSON',
        help='JSON to load arguments from. The value an argument takes is determined by the order: CMDLINE > JSON > DEFAULT'
    )
    parser.add_argument(
        '--outfile-prefix',
        type=str,
        default='box2d-json/GeneratedRaptorWalker',
        help='What to prefix the name of the generated JSON file'
    )
    parser.add_argument(
        '--body-type',
        type=str,
        default='raptor',
        help='What type of body to generate (bipedal, centipede, raptor, dog)'
    )
    parser.add_argument(
        '--num-bodies',
        type=int,
        default=1,
        help='How many bodies to generate'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        default='uniform',
        help='Distribution to sample from. If argument provided, will look at appropriate parameter args'
    )
    parser.add_argument(
        '--hull-density',
        type=float,
        nargs='+',
        default=5.0,
        help='The density of the center hull segment (default 5.0)')
    parser.add_argument(
        '--hull-friction',
        type=float,
        nargs='+',
        default=0.1,
        help='The friction of the center hull segment (default 0.1)')
    parser.add_argument(
        '--head-density',
        type=float,
        nargs='+',
        default=5.0,
        help='The density of the head (default 5.0)')
    parser.add_argument(
        '--head-friction',
        type=float,
        nargs='+',
        default=0.0,
        help='The friction of the head (default 0.0)')
    parser.add_argument(
        '--neck-density',
        type=float,
        nargs='+',
        default=5.0,
        help='The density of neck segments (default 5.0)')
    parser.add_argument(
        '--neck-friction',
        type=float,
        nargs='+',
        default=0.0,
        help='The friction of neck segments (default 0.0)')
    parser.add_argument(
        '--tail-density',
        type=float,
        nargs='+',
        default=5.0,
        help='The density of tail segments (default 5.0)')
    parser.add_argument(
        '--tail-friction',
        type=float,
        nargs='+',
        default=0.0,
        help='The friction of tail segments (default 28.0)')
    parser.add_argument(
        '--leg-density',
        type=float,
        nargs='+',
        default=1.0,
        help='The density of leg segments (default 1.0)')
    parser.add_argument(
        '--leg-friction',
        type=float,
        nargs='+',
        default=0.2,
        help='The friction of leg segments (default 0.2)')
    parser.add_argument(
        '--neck-segments',
        type=int,
        nargs='+',
        default=4,
        help='Number of neck segments (default 4)')
    parser.add_argument(
        '--tail-segments',
        type=int,
        nargs='+',
        default=4,
        help='Number of tail segments (default 4)')
    parser.add_argument(
        '--hull-width',
        type=float,
        nargs='+',
        default=28.0,
        help='The width of the center hull segment (default 28.0)')
    parser.add_argument(
        '--hull-height',
        type=float,
        nargs='+',
        default=28.0,
        help='The height of the center hull segment (default 28.0)')
    parser.add_argument(
        '--head-width',
        type=float,
        nargs='+',
        default=12.5,
        help='Head width (default 12.5)')
    parser.add_argument(
        '--head-height',
        type=float,
        nargs='+',
        default=24.0,
        help='Head height (default 24.0)')
    parser.add_argument(
        '--head-angle',
        type=float,
        nargs='+',
        default=0.52,
        help='Head angle (default 0.52)')
    parser.add_argument(
        '--neck-width',
        type=float,
        nargs='+',
        default=9.0,
        help='Ending neck width, right before head (default 9.0)')
    parser.add_argument(
        '--neck-height',
        type=float,
        nargs='+',
        default=9.0,
        help='Ending neck height, right before head (default 9.0)')
    parser.add_argument(
        '--tail-width',
        type=float,
        nargs='+',
        default=22.0,
        help='Ending tail width (default 22.0)')
    parser.add_argument(
        '--tail-height',
        type=float,
        nargs='+',
        default=4.0,
        help='Ending tail height (default 4.0)')
    parser.add_argument(
        '--thigh-width',
        type=float,
        nargs='+',
        default=10.0,
        help='Thigh (highest up leg segment) width (default 10.0)')
    parser.add_argument(
        '--thigh-height',
        type=float,
        nargs='+',
        default=34.0,
        help='Thigh (highest up leg segment) height (default 34.0)')
    parser.add_argument(
        '--shin-width',
        type=float,
        nargs='+',
        default=7.0,
        help='Shin (second highest leg segment) width (default 7.0)')
    parser.add_argument(
        '--shin-height',
        type=float,
        nargs='+',
        default=24.0,
        help='Shin (second highest leg segment) height (default 24.0)')
    parser.add_argument(
        '--foot-width',
        type=float,
        nargs='+',
        default=6.4,
        help='Foot (third highest leg segment) width (default 6.4)')
    parser.add_argument(
        '--foot-height',
        type=float,
        nargs='+',
        default=20.0,
        help='Foot (third highest leg segment) height (default 20.0)')
    parser.add_argument(
        '--toes-width',
        type=float,
        nargs='+',
        default=5.6,
        help='Toes (bottom leg segment) width (default 5.6)')
    parser.add_argument(
        '--toes-height',
        type=float,
        nargs='+',
        default=16.0,
        help='Toes (bottom leg segment) height (default 16.0)')

    for k in additional_parser_args.keys():
        parser.add_argument(
            additional_parser_args[k]['name'],
            type=additional_parser_args[k]['type'],
            metavar=additional_parser_args[k]['metavar'],
            help=additional_parser_args[k]['help']
        )
    return vars(parser.parse_args())

def parse_json_args(jsonfn):
    try:
        with open(jsonfn, 'r') as jf:
            args = json.load(jf)
        return args
    except:
        return {}

def parse_default_args(additional_default_args={}):
    default_args = {
        'json_file' : '',
        'outfile_prefix' : 'box2d-json/GeneratedRaptorWalker',
        'body_type' : 'raptor',
        'num_bodies' : 1,
        'distribution' : 'uniform',
        'hull_density' : 5.0,
        'hull_friction' : 0.1,
        'head_density' : 5.0,
        'head_friction' : 0.0,
        'neck_density' : 5.0,
        'neck_friction' : 0.0,
        'tail_density' : 5.0,
        'tail_friction' : 0.0,
        'leg_density' : 1.0,
        'leg_friction' : 0.2,
        'neck_segments' : 4,
        'tail_segments' : 4,
        'hull_width' : 28.0,
        'hull_height' : 28.0,
        'head_width' : 12.5,
        'head_height' : 24.0,
        'head_angle' : 0.52,
        'neck_width' : 9.0,
        'neck_height' : 9.0,
        'tail_width' : 22.0,
        'tail_height' : 4.0,
        'thigh_width' : 10.0,
        'thigh_height' : 34.0,
        'shin_width' : 7.0,
        'shin_height' : 24.0,
        'foot_width' : 6.4,
        'foot_height' : 20.0,
        'toes_width' : 5.6,
        'toes_height' : 16.0,
    }
    for k in additional_default_args.keys():
        if k not in default_args:
            default_args[k] = additional_default_args[k]
    return default_args


def parse_args(additional_parser_args={}, additional_default_args={}):
    # Load arguments from command line
    cmdline_args = parse_cmdline_args(
        additional_parser_args=additional_parser_args)

    # Load arguments from JSON, if it was supplied
    json_args = parse_json_args(cmdline_args['json_file'])

    # Load default arguments
    default_args = parse_default_args(
        additional_default_args=additional_default_args)

    # Overwrite arguments in the default args if they were supplied
    # Order of Overwriting: CMDLINE > JSON > DEFAULT
    args = copy.deepcopy(default_args)
    for k in default_args.keys():
        if cmdline_args[k] is not None:
            args[k] = cmdline_args[k]
        elif k in json_args and json_args[k] is not None:
            args[k] = json_args[k]
    return args