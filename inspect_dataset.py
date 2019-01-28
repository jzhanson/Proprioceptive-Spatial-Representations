import os
import copy
import numpy as np
from args import parse_args
import json
import re
import time

import gym
from common.environment import create_env

from visualize import Visualize

def inspect_dataset(args):
    for root, directories, filenames in os.walk(args['directory']):
        for filename in filenames:
            if '.json' not in filename:
                continue
            filepath = os.path.join(root, filename)
            # TODO(josh): read metafile and use info for something
            '''
            with open(args['metafile_directory'] + '/info.meta') as metafile:
                info = metafile.read()
                if 'BipedalWalker' in info:
                    body_type = 'BipedalWalker'
                elif 'RaptorWalker' in info:
                    body_type = 'RaptorWalker'
                elif 'CentipedeWalker' in info:
                    body_type = 'CentipedeWalker'
            '''
            # TODO(josh): make inspect_dataset support multiple body types in
            # the same dataset
            body_type = args['body_type']

            with open(filepath) as json_file:
                jsondata = json.load(json_file)
            num_segments = 0
            hull_position = 0
            reporting_bodies = []
            reporting_joints = []
            for k in jsondata.keys():
                if body_type == 'BipedalWalker':
                    body_re = re.compile('((Body\d+)|(Hull))$')
                    if body_re.match(k):
                        if k == 'Hull':
                            hull_position = num_segments
                        num_segments += 1
                    # Reporting segments
                    fixture_re = re.compile('.*Fixture$')
                    joint_re = re.compile('.*Joint')
                    if fixture_re.match(k) is None:
                        if joint_re.match(k):
                            if jsondata[k]['ReportState']:
                                reporting_joints.append(k)
                        else:   # Body or leg
                            if jsondata[k]['ReportState']:
                                reporting_bodies.append(k)
                elif body_type == 'RaptorWalker':
                    neck_segments = 0
                    tail_segments = 0
                    has_head = False
                    body_re = re.compile('((Neck\d+)|(Tail\d+)|Head|Hull)$')
                    # neck_segments doesn't include head
                    neck_re = re.compile('Neck\d+$')
                    tail_re = re.compile('Tail\d+$')
                    head_re = re.compile('Head')
                    if body_re.match(k):
                        num_segments += 1
                    if neck_re.match(k):
                        neck_segments += 1
                    if tail_re.match(k):
                        tail_segments += 1
                    if head_re.match(k):
                        has_head = True
                    # Reporting segments
                    fixture_re = re.compile('.*Fixture$')
                    joint_re = re.compile('Joint.*')
                    if fixture_re.match(k) is None:
                        if joint_re.match(k):
                            if jsondata[k]['ReportState']:
                                reporting_joints.append(k)
                        else:   # Body or leg
                            if jsondata[k]['ReportState']:
                                reporting_bodies.append(k)

            has_odd_segments = (num_segments + 1) % 2
            off_center_hull = (hull_position != num_segments // 2 - has_odd_segments)

            print(filepath)
            print('num_segments : ' + str(num_segments))
            if body_type == 'BipedalWalker':
                print('reporting bodies : ' + str(reporting_bodies))
                print('reporting joints : ' + str(reporting_joints))
                print('off center hull : ' + str(off_center_hull))
                print('hull_position : ' + str(hull_position))
            if body_type == 'RaptorWalker':
                print('reporting bodies : ' + str(reporting_bodies))
                print('reporting joints : ' + str(reporting_joints))
                print('neck segments : ' + str(neck_segments))
                print('tail segments : ' + str(tail_segments))
                print('has head : ' + str(has_head))
            print('\n')

            if args['display_bodies']:
                # Note that args don't really matter
                args['env'] = 'JSONWalker-' + filepath
                visualize = Visualize(args)
                visualize.main()

if __name__=='__main__':
    args = parse_args(
        additional_parser_args={
            'display_bodies' : {
                'name' : '--display-bodies',
                'metavar' : 'DB',
                'help' : 'whether to display bodies'
                },
            'directory' : {
                'name' : '--directory',
                'type' : str,
                'metavar' : 'DIR',
                'help' : 'Directory with dataset to inspect'
                },
            'body_type' : {
                'name' : '--body-type',
                'type' : str,
                'metavar' : 'BT',
                'help' : 'Body type of JSONs (BipedalWalker, RaptorWalker, etc'
                },
            },
        additional_default_args={
            'display_bodies' : True,
            'directory' : '',
            'body_type' : 'BipedalWalker'
            }
        )

    inspect_dataset(args)
