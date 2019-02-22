import os
from evaluate import evaluate
import copy
import numpy as np
import torch
from args import parse_args
import time

def directory_evaluate(args):
    start_time = time.time()
    all_evaluation_statistics = {}

    files_list = [f for f in os.listdir(args['json_directory']) if '.json' in f]
    for f in files_list:
        new_args = copy.deepcopy(args)
        new_args['env'] = 'JSONWalker-' + args['json_directory'] + '/' + f
        evaluation_statistics = evaluate(new_args)
        all_evaluation_statistics[new_args['env']] = evaluation_statistics

    for k in all_evaluation_statistics.keys():
        all_episode_returns = all_evaluation_statistics[k]['all_episode_returns']
        all_episode_successes = all_evaluation_statistics[k]['all_episode_successes']

        print('Environment: '+k)
        print('\tAverage Episodic Return: \n\t\tmean: {0}\n\t\tstd: {1}\n\t\t  \
                min: {2}\n\t\tmax: {3}'.format(
                    np.mean(all_episode_returns),
                    np.std(all_episode_returns),
                    np.min(all_episode_returns),
                    np.max(all_episode_returns)))
        print('\tAverage Episodic Success: \n\t\tmean: {0} ({1}/{2})\n\t\t     \
                std: {3}\n\t\tmin: {4}\n\t\tmax: {5}'.format(
                    np.mean(all_episode_successes),
                    np.sum(all_episode_successes),
                    all_evaluation_statistics[k]['Number Total'],
                    np.std(all_episode_successes),
                    np.min(all_episode_successes),
                    np.max(all_episode_successes)))

    # Save all evaluation statistics
    output_path = os.path.join(os.path.dirname(args['load_file']),
                               args['output_directory'],
                              'JSONWalker-'+(args['json_directory'].replace('/',
                                  '-'))+'-evaluation-statistics-evalep{}.pth'
                              .format(args['num_episodes']))
    torch.save({
        'all_evaluation_statistics' : all_evaluation_statistics,
    }, output_path)

    end_time = time.time()
    total_episodes = len(files_list) * args['num_episodes']
    print('directory evaluate total time for %d episodes: %d' % (total_episodes,
        end_time - start_time))
    print('directory evaluate overall seconds per episode: %f' %
        ((end_time - start_time) / total_episodes))
    return all_evaluation_statistics


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
            'json_directory' : {
                'name' : '--json-directory',
                'type' : str,
                'metavar' : 'JD',
                'help' : 'Directory with JSONS run evaluate on'
            },
            'output_directory' : {
                'name' : '--output-directory',
                'type' : str,
                'metavar' : 'OD',
                'help' : 'Directory to group outputs in'
            },
            'load_directory' : {
                'name' : '--load-directory',
                'type' : str,
                'metavar' : 'LD',
                'help' : 'Directory to load model files from'
            },
            # models_start and models_end are both inclusive
            'models_start' : {
                'name' : '--models-start',
                'type' : int,
                'metavar' : 'MS',
                'help' : 'First model number to test'
            },
            'models_end' : {
                'name' : '--models-end',
                'type' : int,
                'metavar' : 'ME',
                'help' : 'Last model number to test'
            },
            'models_step' : {
                'name' : '--models-step',
                'type' : int,
                'metavar' : 'MS',
                'help' : 'How much the model number increases'
            },
            'evaluate_all' : {
                'name' : '--evaluate-all',
                'type' : bool,
                'metavar' : 'EA',
                'help' : 'Whether or not to evaluate all .pth files in load_directory'
            }
        },
        additional_default_args={
            'num_episodes' : 100,
            'render_video' : False,
            'json_directory' : '',
            'output_directory' : '',
            'load_directory' : '',
            'models_start' : -1,
            'models_end' : -1,
            'models_step' : -1,
            'evaluate_all' : False
        }
    )

    all_model_statistics = []

    # We want the directory hierarchy to be: model -> dataset -> checkpoint
    if args['models_start'] != -1:
        assert False
        for i in range(args['models_start'], args['models_end'] + args['models_step'], args['models_step']):
            current_args = copy.deepcopy(args)
            # Provide --load-directory instead of --load-file if going to iterate through files
            current_args['load_file'] = os.path.join(args['load_directory'], 'model.' + str(i) + '.pth')
            # Note: Make the input output_directory be the dataset name
            current_args['output_directory'] = os.path.join(args['output_directory'], str(i))
            current_model_statistics = directory_evaluate(current_args)
            all_model_statistics.append(current_model_statistics)


    # TODO(josh): no guarantee that the checkpoints will be read low -> high, so appending doesn't make sense. Also, edit this block and the above block to use multiprocessing
    elif args['evaluate_all']:
        assert False
        models_list = [f for f in os.listdir(args['load_directory']) if '.pth' in f]
        for model in models_list:
            print(model)
            current_args = copy.deepcopy(args)
            current_args['load_file'] = os.path.join(args['load_directory'], model)
            # Note: Make the input output_directory be the dataset name
            current_args['output_directory'] = os.path.join(args['output_directory'], model)
            current_model_statistics = directory_evaluate(current_args)
            all_model_statistics.append(current_model_statistics)

    else:
        current_args = copy.deepcopy(args)
        #current_args['output_directory'] = os.path.join(args['output_directory'], os.path.split(args['load_file'])[1])
        directory_evaluate(args)

