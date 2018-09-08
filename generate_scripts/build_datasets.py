import sys
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
            self.dataset_split = np.random.permutation(range(self.args['num_segments'][0], self.args['num_segments'][1] + 1))
        else:
            self.dataset_split = self.args['num_segments']
        print(self.dataset_split)

    def build_dataset(self, dataset):
        args = self.args
        args['directory'] = self.directory + '-' + dataset

        # Assume even # of dataset splits
        # TODO(josh): add ways to split datasets on things besides number of body segments
        if dataset == 'train':
            if type(self.dataset_split) is list:
                for i in range(self.args['num_bodies']):
                    args['num_segments'] = random.choice(self.dataset_split[0:(len(self.dataset_split) // 2)])
                    args['num_bodies'] = 1
                    randomize = RandomizeBodies(self.body_type, args)
                    randomize.build_bodies()
            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies()

        elif dataset == 'valid':
            for i in range(self.args['num_bodies']):
                args['num_segments'] = random.choice(self.dataset_split[len(self.dataset_split) // 2:len(self.dataset_split) // 2 + len(self.dataset_split) // 4])
                args['num_bodies'] = 1
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies()
            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies()
        elif dataset == 'test':
            for i in range(self.args['num_bodies']):
                args['num_segments'] = random.choice(self.dataset_split[len(self.dataset_split) // 2 + len(self.dataset_split) // 4:])
                args['num_bodies'] = 1
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies()
            else:
                randomize = RandomizeBodies(self.body_type, args)
                randomize.build_bodies()

if __name__ == '__main__':
    body_type = sys.argv[1] if len(sys.argv) > 1 else 'BipedalWalker'
    args = parse_args(body_type)

    build = BuildDatasets(body_type, args)
    build.build_dataset('train')
    build.build_dataset('valid')
    build.build_dataset('test')
