import os
import json
import argparse
import random
from generate_raptor import GenerateRaptor
from randomize_args import parse_args

class RandomizeBodies:
    def __init__(self, args):
        self.args = args

    def build_bodies(self):
        for i in range(self.args['num_bodies']):
            gen_args = {}
            # TODO(josh): add these arguments to randomize_args
            gen_args['rigid_spine'] = False
            gen_args['spine_motors'] = True
            for k in self.args.keys():
                if k not in ['outfile_prefix', 'num_bodies', 'distribution']:
                    # No distribution params provided
                    if type(self.args[k]) is float or type(self.args[k]) is int:
                        gen_args[k] = self.args[k]
                    # Distribution parameters provided
                    elif type(self.args[k]) is list:
                        if self.args['distribution'] == 'uniform':
                            lo = self.args[k][0]
                            hi = self.args[k][1]
                            if type(lo) is int:
                                gen_args[k] = random.randint(lo, hi)
                            else:
                                gen_args[k] = random.uniform(lo, hi)

            gen = GenerateRaptor(gen_args)

            gen.build()
            print(self.args)
            gen.write_to_json(filename=self.args['outfile_prefix'] + str(i) + '.json')


if __name__ == '__main__':
    args = parse_args()

    randomize = RandomizeBodies(args)

    randomize.build_bodies()


