import os
from evaluate import evaluate
import copy
from args import parse_args

def directory_evaluate(args):
    files_list = [f for f in os.listdir(args['directory']) if '.json' in f]
    for f in files_list:
        new_args = copy.deepcopy(args)
        new_args['env'] = 'JSONWalker-' + args['directory'] + '/' + f
        evaluate(new_args)


if __name__=='__main__':
    args = parse_args(
        additional_parser_args={
            'num_episodes' : {
                'name' : '--num-episodes',
                'type' : int,
                'metavar' : 'NE',
                'help' : 'how many epiosdes in evaluation (default: 100)'
            },
            'render_video' : {
                'name' : '--render-video',
                'metavar' : 'RV',
                'help' : 'whether to render evaluation episodes'
            },
            'directory' : {
                'name' : '--directory',
                'type' : str,
                'metavar' : 'DIR',
                'help' : 'Directory with JSONS run evaluate on'
            }
        },
        additional_default_args={
            'num_episodes' : 100,
            'render_video' : False,
            'directory' : ''
        }
    )

    directory_evaluate(args)

