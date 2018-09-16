import copy
import sys, os
import random
import numpy as np
from randomize_bodies import RandomizeBodies
from randomize_args import parse_args

class BuildDatasets:
    def __init__(self, body_type, args):
        self.args = args
        self.directory = self.args['directory']
        self.body_type = body_type
        if type(self.args['num_segments']) is list:
            self.dataset_split = list(np.random.permutation(range(self.args['num_segments'][0], self.args['num_segments'][1] + 1)))
        else:
            self.dataset_split = self.args['num_segments']
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        self.metafile = open(self.directory + '/info.meta', 'w+')
        self.metafile.write('Dataset split: ' + str(self.dataset_split) + '\n')
        print(self.dataset_split)

    def build_dataset(self, dataset):
        args = copy.deepcopy(self.args)
        args['directory'] = self.directory + '/' + dataset

        self.metafile.write(dataset + '\n')

        num_training_segment_configs = len(self.dataset_split) // 2
        num_validation_segment_configs = len(self.dataset_split) // 4
        num_testing_segment_configs = len(self.dataset_split) // 4

        # Assume even # of dataset splits
        # TODO(josh): add ways to split datasets on things besides number of body segments
        if dataset == 'train':
            if type(self.dataset_split) is list:
                for i in range(0, num_training_segment_configs):
                    args['num_segments'] = self.dataset_split[i]
                    args['num_bodies'] = self.args['num_bodies']
                    args['outfile_prefix'] = self.args['outfile_prefix'] + str(args['num_segments']) + 'segments-'
                    randomize = RandomizeBodies(self.body_type, args)
                    randomize.build_bodies(self.metafile)
            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies(self.metafile)

        elif dataset == 'valid':
            if type(self.dataset_split) is list:
                for i in range(num_training_segment_configs, num_training_segment_configs+num_validation_segment_configs):
                    args['num_segments'] = self.dataset_split[i]
                    args['num_bodies'] = self.args['num_bodies']
                    args['outfile_prefix'] = self.args['outfile_prefix'] + str(args['num_segments']) + 'segments-'
                    randomize = RandomizeBodies(self.body_type, args)
                    randomize.build_bodies(self.metafile)
            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies(self.metafile)
        elif dataset == 'test':
            if type(self.dataset_split) is list:
                for i in range(num_training_segment_configs+num_validation_segment_configs, len(self.dataset_split)):
                    args['num_segments'] = self.dataset_split[i]
                    args['num_bodies'] = self.args['num_bodies']
                    args['outfile_prefix'] = self.args['outfile_prefix'] + str(args['num_segments']) + 'segments-'
                    randomize = RandomizeBodies(self.body_type, args)
                    randomize.build_bodies(self.metafile)

            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies(self.metafile)

if __name__ == '__main__':
    body_type = sys.argv[1] if len(sys.argv) > 1 else 'BipedalWalker'
    args = parse_args(body_type)

    build = BuildDatasets(body_type, args)
    build.build_dataset('train')
    build.build_dataset('valid')
    build.build_dataset('test')
