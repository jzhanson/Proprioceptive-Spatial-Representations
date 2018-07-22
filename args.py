from __future__ import print_function, division
import os
import copy
import json
import argparse
import importlib


def parse_cmdline_args(additional_parser_args={}):
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument(
        '--json_file',
        type=str,
        default='',
        metavar='JSON',
        help='JSON to load arguments from. The value an argument takes is determined by the order: CMDLINE > JSON > DEFAULT'
    )
    parser.add_argument(
        '--lr',
        type=float,
        metavar='LR',
        help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--gamma',
        type=float,
        metavar='G',
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--tau',
        type=float,
        metavar='T',
        help='parameter for GAE (default: 1.00)')
    parser.add_argument(
        '--seed',
        type=int,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--workers',
        type=int,
        metavar='W',
        help='how many training processes to use (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        metavar='NS',
        help='number of forward steps in A3C (default: 300)')
    parser.add_argument(
        '--max-episode-length',
        type=int,
        metavar='M',
        help='maximum length of an episode (default: 10000)')
    parser.add_argument(
        '--env',
        metavar='ENV',
        help='environment to train on (default: BipedalWalker-v2)')
    parser.add_argument(
        '--shared-optimizer',
        metavar='SO',
        help='use an optimizer without shared statistics.')
    parser.add_argument(
        '--save-directory',
        metavar='SD',
        help='where to save model files and logs.')
    parser.add_argument(
        '--load-file',
        metavar='LF',
        help='pth file to load a training checkpoint.')
    parser.add_argument(
        '--load-best',
        metavar='LB',
        help='whether to load the best model/optimizer from a checkpoint.')
    parser.add_argument(
        '--save-max',
        metavar='SM',
        help='Save model on every test run high score matched or bested')
    parser.add_argument(
        '--optimizer',
        metavar='OPT',
        help='shares optimizer choice of Adam or RMSprop')
    parser.add_argument(
        '--model-name',
        metavar='M',
        help='Model type to use')
    parser.add_argument(
        '--stack-frames',
        type=int,
        metavar='SF',
        help='Choose number of observations to stack')
    parser.add_argument(
        '--grid-edge',
        type=int,
        metavar='GE',
        help='grid size')
    parser.add_argument(
        '--grid-scale',
        type=float,
        help='grid scale')
    parser.add_argument(
        '--blur-frames',
        type=int,
        metavar='BF',
        help='Choose number of frames to blur')
    parser.add_argument(
        '--blur-discount',
        type=float,
        metavar='BD',
        help='Choose discount factor for motion blur')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--amsgrad',
        metavar='AM',
        help='Adam optimizer amsgrad parameter')
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
    # A3G training args
    default_args = {
        'json_file' : '',
        'lr' : 0.0001,
        'gamma' : 0.99,
        'tau' : 1.00,
        'seed' : 1,
        'workers' : 32,
        'num_steps' : 20,
        'max_episode_length' : 10000,
        'env' : 'BipedalWalker-v2',
        'shared_optimizer' : True,
        'load_file' : '',
        'load_best' : False,
        'save_max' : True,
        'optimizer' : 'Adam',
        'save_directory' : 'saved/default/',
        'model_name' : 'models.mlp',
        'stack_frames' : 1,
        'grid_edge' : 16,
        'grid_scale' : 5.44,
        'blur_frames' : 1,
        'blur_discount' : 1.0,
        'gpu_ids' : [-1],
        'amsgrad' : True,
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
