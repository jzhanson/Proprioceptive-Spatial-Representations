import os, sys
import json
import argparse
import random
from generate_bipedal import GenerateBipedal
from generate_centipede import GenerateCentipede
from generate_raptor import GenerateRaptor
#from generate_dog import GenerateDog
from randomize_args import parse_args

class RandomizeBodies:
    def __init__(self, body_type, args):
        self.body_type = body_type
        self.args = args
        if not os.path.exists(self.args['directory']):
            os.mkdir(self.args['directory'])

    def build_bodies(self, metafile=None):
        for i in range(self.args['num_bodies']):
            gen_args = {}
            gen_args['filename'] = self.args['directory'] + '/' + self.args['outfile_prefix'] + str(i) + '.json'
            for k in self.args.keys():
                if k not in ['outfile_prefix', 'num_bodies', 'distribution']:
                    # Distribution parameters provided
                    if type(self.args[k]) is list:
                        if self.args['distribution'] == 'uniform':
                            lo = self.args[k][0]
                            hi = self.args[k][1]
                            if type(lo) is int:
                                gen_args[k] = random.randint(lo, hi)
                            else:
                                gen_args[k] = random.uniform(lo, hi)
                    # No distribution parameters provided or boolean parameter
                    else:
                        gen_args[k] = self.args[k]

            if metafile is not None:
                metafile.write(gen_args['filename'] + '\n' + str(gen_args) + '\n\n')

            if self.body_type == 'BipedalWalker':
                gen = GenerateBipedal(gen_args)
            elif self.body_type == 'CentipedeWalker':
                gen = GenerateCentipede(gen_args)
            elif self.body_type == 'RaptorWalker':
                gen = GenerateRaptor(gen_args)
            #elif self.body_type == 'DogWalker':
            #    gen = GenerateDog(gen_args)

            gen.build()

            gen.write_to_json()


if __name__ == '__main__':
    # TODO(josh): figure out a way to avoid double-parsing arguments (but we need to know which body type it is before we parse arguments because some attributes have different defaults)
    # The first command line argument is required
    body_type = sys.argv[1] if len(sys.argv) > 1 else 'BipedalWalker'
    args = parse_args(body_type)

    randomize = RandomizeBodies(body_type, args)

    randomize.build_bodies()


