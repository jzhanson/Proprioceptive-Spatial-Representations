import os
from evaluate import evaluate
import copy
import numpy as np
from args import parse_args

def directory_evaluate(args):
    all_evaluation_statistics = {}

    files_list = [f for f in os.listdir(args['directory']) if '.json' in f]
    for f in files_list:
        new_args = copy.deepcopy(args)
        new_args['env'] = 'JSONWalker-' + args['directory'] + '/' + f
        evaluation_statistics = evaluate(new_args)
        all_evaluation_statistics[new_args['env']] = evaluation_statistics

    for k in all_evaluation_statistics.keys():
        all_episode_returns = all_evaluation_statistics[k]['all_episode_returns']
        all_episode_successes = all_evaluation_statistics[k]['all_episode_successes']

        print('Environment: '+k)
        print('\tAverage Episodic Return: \n\t\tmean: {0}\n\t\tstd: {1}\n\t\tmin: {2}\n\t\tmax: {3}'.format(
            np.mean(all_episode_returns), np.std(all_episode_returns),
            np.min(all_episode_returns), np.max(all_episode_returns)))
        print('\tAverage Episodic Success: \n\t\tmean: {0} ({1}/{2})\n\t\tstd: {3}\n\t\tmin: {4}\n\t\tmax: {5}'.format(
            np.mean(all_episode_successes), np.sum(all_episode_successes), all_evaluation_statistics[k]['Number Total'],
            np.std(all_episode_successes), np.min(all_episode_successes), np.max(all_episode_successes)))


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

