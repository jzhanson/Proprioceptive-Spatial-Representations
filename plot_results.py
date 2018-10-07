import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from common.stat_utils import smooth
from args import parse_args

# TODO(josh): build ability to make graphs for each JSON
def plot_statistics(all_model_statistics, graphs_directory):
    # Assumes model filenames are of the form model.x.pth
    checkpoints = []
    all_returns = []
    all_successes = []
    # TODO(josh): extend to get median statistics
    for checkpoint_name in all_model_statistics.keys():
        num_jsons = len(all_model_statistics[checkpoint_name].keys())
        if num_jsons == 0:
            print(checkpoint_name)
            continue
        # Currently, a checkpoint is the mean of means (over JSONs, over episodes)
        checkpoints.append(int(checkpoint_name.split('.')[1]))
        returns_total_means = 0
        successes_total_means = 0
        for json_name in all_model_statistics[checkpoint_name].keys():
            returns_total_means += np.mean(all_model_statistics[checkpoint_name][json_name]['all_episode_returns'])
            successes_total_means += np.mean(all_model_statistics[checkpoint_name][json_name]['all_episode_successes'])
        all_returns.append(returns_total_means / num_jsons)
        all_successes.append(successes_total_means / num_jsons)
    sorted_checkpoints, sorted_all_returns = zip(*sorted(zip(checkpoints, all_returns)))

    # Raw plot
    plt.clf()
    plt.plot(sorted_checkpoints, sorted_all_returns)
    plt.title('Directory Evaluation Returns')
    plt.xlabel('Model checkpoint')
    plt.ylabel('Average reward per episode')
    # TODO(josh): more descriptive saved graph names
    plt.savefig(os.path.join(graphs_directory, 'evaluate_returns.png'))
    plt.savefig(os.path.join(graphs_directory, 'evaluate_returns.eps'))

    # Smoothed version
    plt.clf()
    all_returns_smooth = smooth(np.array(sorted_all_returns), np.array(sorted_checkpoints))
    plt.plot(sorted_checkpoints, all_returns_smooth, 'k', color='#CC4F1B')
    plt.savefig(os.path.join(graphs_directory, 'evaluate_returns_smooth.png'))
    plt.savefig(os.path.join(graphs_directory, 'evaluate_returns_smooth.eps'))


if __name__=='__main__':
    args = parse_args(
        additional_parser_args={
            'model_directory' : {
                'name' : '--model-directory',
                'type' : str,
                'metavar' : 'MD',
                'help' : 'Which parent directory to look in'
            },
            'evaluation_prefix' : {
                'name' : '--evaluation-prefix',
                'type' : str,
                'metavar' : 'EP',
                'help' : 'The prefix of each evaluation directory (not including last model.x.pth)'
            },
            'graphs_directory' : {
                'name' : '--graphs-directory',
                'type' : str,
                'metavar' : 'GD',
                'help' : 'The directory in which to save the graphs'
            }
        },
        additional_default_args={
            'model_directory' : '',
            'evaluation_prefix' : '',
            'graphs_directory' : ''
        }
    )

    # Directories go model -> checkpoints -> JSON -> evaluation_statistics.pth
    models_list = [f for f in os.listdir(args['model_directory']) if not os.path.isdir(os.path.join(args['model_directory'], f)) and '.pth' in f]

    # model_statistics is (checkpoint, (json, (all_episode_returns | all_episode_successes, np array))
    all_model_statistics = {}
    for model in models_list:
        current_model_statistics = {}
        current_evaluation_directory = os.path.join(args['model_directory'], args['evaluation_prefix'] + model)

        for json_subdir in [f for f in os.listdir(current_evaluation_directory) if os.path.isdir(os.path.join(current_evaluation_directory, f))]:
            current_evaluation_statistics_path = os.path.join(current_evaluation_directory, json_subdir, 'evaluation_statistics.pth')

            # Makes the assumption that the json directory is [json filename]+junk
            json_name = json_subdir.split('.json')[0]
            if os.path.isfile(current_evaluation_statistics_path):
                json_evaluation_statistics = torch.load(current_evaluation_statistics_path)
                current_model_statistics[json_name] = json_evaluation_statistics

        all_model_statistics[model] = current_model_statistics

    plot_statistics(all_model_statistics, args['graphs_directory'])



