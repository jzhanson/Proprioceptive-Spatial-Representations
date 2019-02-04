import copy
import sys, os
import math
import random
import numpy as np
from randomize_bodies import RandomizeBodies
from randomize_args import parse_args

class BuildDatasets:
    def __init__(self, body_type, args):
        self.args = args
        self.directory = self.args['directory']
        self.body_type = body_type
        if self.body_type == 'BipedalWalker':
            if type(self.args['num_segments']) is list:
                self.dataset_split = list(np.random.permutation(range(self.args['num_segments'][0],
                    self.args['num_segments'][1] + 1)))
            else:
                self.dataset_split = self.args['num_segments']
        # For raptor, dataset_split will be a tuple of neck_segments, tail-segments
        elif self.body_type == 'RaptorWalker':
            # If both neck_segments and tail_segments are constant, just generate 2/1/1 split
            # TODO(josh): make this support different range sizes for neck/tail segments
            if type(self.args['neck_segments']) is not list \
                and type(self.args['tail_segments']) is not list:
                neck_split = [self.args['neck_segments'] for i in range(4)]
                tail_split = [self.args['tail_segments'] for i in range(4)]

            elif type(self.args['neck_segments']) is list   \
                and type(self.args['tail_segments']) is list:
                neck_range = range(self.args['neck_segments'][0],
                    self.args['neck_segments'][1] + 1)
                tail_range = range(self.args['tail_segments'][0],
                    self.args['tail_segments'][1] + 1)

                neck_split = list(np.random.permutation(neck_range))
                tail_split = list(np.random.permutation(tail_range))

            elif type(self.args['neck_segments']) is not list   \
                and type(self.args['tail_segments']) is list:
                tail_range = range(self.args['tail_segments'][0],
                    self.args['tail_segments'][1] + 1)

                neck_split = [self.args['neck_segments'] for i in tail_range]
                tail_split = list(np.random.permutation(tail_range))

            elif type(self.args['neck_segments']) is list   \
                and type(self.args['tail_segments']) is not list:
                neck_range = range(self.args['neck_segments'][0],
                    self.args['neck_segments'][1] + 1)

                neck_split = list(np.random.permutation(neck_range))
                tail_split = [self.args['tail_segments'] for i in neck_range]

            self.dataset_split = list(zip(neck_split, tail_split))

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        self.metafile = open(self.directory + '/info.meta', 'w+')
        self.metafile.write('Dataset split: ' + str(self.dataset_split) + '\n')
        print(self.dataset_split)

    def build_dataset(self, dataset):
        args = copy.deepcopy(self.args)
        args['directory'] = self.directory + '/' + dataset

        self.metafile.write(dataset + '\n')

        num_to_split = self.args['num_bodies']
        if type(self.dataset_split) is list:
            num_to_split = len(self.dataset_split)

        # If we have an odd number of segments, training configs will be
        # half rounded down
        num_training_configs = num_to_split // 2
        # Validation will be half of what's left rounded up
        num_validation_configs = math.ceil((num_to_split
            - num_training_configs) / 2.)
        num_testing_configs = num_to_split - num_training_configs\
            - num_validation_configs

        # Assume even # of dataset splits
        # TODO(josh): add ways to split datasets on things besides number of body segments
        if dataset == 'train':
            if type(self.dataset_split) is list:
                for i in range(0, num_training_configs):
                    if self.body_type == 'BipedalWalker':
                        args['num_segments'] = self.dataset_split[i]
                        args['outfile_prefix'] = self.args['outfile_prefix']   \
                            + '-' + str(args['num_segments']) + 'segments-'
                    elif self.body_type == 'RaptorWalker':
                        args['neck_segments'] = self.dataset_split[i][0]
                        args['tail_segments'] = self.dataset_split[i][1]
                        args['outfile_prefix'] = self.args['outfile_prefix'] +  \
                            '-batch' + str(i) + '-' + str(args['neck_segments']) + 'neck' +   \
                            str(args['tail_segments']) + 'tail' + 'segments-'

                    args['num_bodies'] = self.args['num_bodies']
            randomize = RandomizeBodies(self.body_type, args)
            randomize.build_bodies(self.metafile)

        elif dataset == 'valid':
            if type(self.dataset_split) is list:
                for i in range(num_training_configs, num_training_configs+num_validation_configs):
                    if self.body_type == 'BipedalWalker':
                        args['num_segments'] = self.dataset_split[i]
                        args['outfile_prefix'] = self.args['outfile_prefix']   \
                            + '-' + str(args['num_segments']) + 'segments-'
                    elif self.body_type == 'RaptorWalker':
                        args['neck_segments'] = self.dataset_split[i][0]
                        args['tail_segments'] = self.dataset_split[i][1]
                        args['outfile_prefix'] = self.args['outfile_prefix'] +  \
                            '-batch' + str(i) + '-' + str(args['neck_segments']) + 'neck' +   \
                            str(args['tail_segments']) + 'tail' + 'segments-'

                    args['num_bodies'] = self.args['num_bodies']
            randomize = RandomizeBodies(self.body_type, args)
            randomize.build_bodies(self.metafile)
        elif dataset == 'test':
            if type(self.dataset_split) is list:
                for i in range(num_training_configs+num_validation_configs, len(self.dataset_split)):
                    if self.body_type == 'BipedalWalker':
                        args['num_segments'] = self.dataset_split[i]
                        args['outfile_prefix'] = self.args['outfile_prefix']   \
                            + '-' + str(args['num_segments']) + 'segments-'
                    elif self.body_type == 'RaptorWalker':
                        args['neck_segments'] = self.dataset_split[i][0]
                        args['tail_segments'] = self.dataset_split[i][1]
                        args['outfile_prefix'] = self.args['outfile_prefix'] +  \
                            '-batch' + str(i) + '-' + str(args['neck_segments']) + 'neck' +   \
                            str(args['tail_segments']) + 'tail' + 'segments-'
                    args['num_bodies'] = self.args['num_bodies']

            randomize = RandomizeBodies(self.body_type, args)
            randomize.build_bodies(self.metafile)

if __name__ == '__main__':
    body_type = sys.argv[1] if len(sys.argv) > 1 else 'BipedalWalker'
    args = parse_args(body_type)

    build = BuildDatasets(body_type, args)
    build.build_dataset('train')
    build.build_dataset('valid')
    build.build_dataset('test')
