import os
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re

from common.stat_utils import smooth
from args import parse_args

def graph(sorted_checkpoints, sorted_means, sorted_stdevs, error_bar_stdev,
    returns_or_successes, save_path):
    # Raw plot
    plt.clf()
    plt.plot(sorted_checkpoints, sorted_means, '.-')
    if error_bar_stdev > 0:
        errors = np.array([error_bar_stdev * sd for sd in sorted_stdevs])
        plt.fill_between(sorted_checkpoints, np.array(sorted_means) - errors,
            np.array(sorted_means) + errors, color='#55A8E2')
        #plt.errorbar(sorted_checkpoints, sorted_all_returns_means,
        #    yerr=[error_bar_stdev * sd for sd in sorted_all_returns_stdevs], fmt='.-')
    plt.title('Directory Evaluation ' + returns_or_successes.title())
    plt.xlabel('Model checkpoint')
    if returns_or_successes == 'returns':
        plt.ylabel('Average reward per episode')
    elif returns_or_successes == 'successes':
        plt.ylabel('Proportion of successful episodes')

    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.eps')

    # Smoothed version
    plt.clf()
    means_smooth = smooth(np.array(sorted_means), np.array(sorted_checkpoints))

    plt.plot(sorted_checkpoints, means_smooth, '.-', color='#CC4F1B')
    if error_bar_stdev > 0:
        errors = np.array([error_bar_stdev * sd for sd in sorted_stdevs])
        plt.fill_between(sorted_checkpoints, np.array(means_smooth) - errors,
            np.array(means_smooth) + errors, color='#EA8A61')

        #plt.errorbar(sorted_checkpoints, all_returns_smooth,
        #    yerr=[error_bar_stdev * sd for sd in sorted_all_returns_stdevs],
        #    fmt='.-', color='#CC4F1B')
    plt.title('Directory Evaluation Returns')
    plt.xlabel('Model checkpoint')
    if returns_or_successes == 'returns':
        plt.ylabel('Average reward per episode')
    elif returns_or_successes == 'successes':
        plt.ylabel('Proportion of successful episodes')
    plt.savefig(save_path + '_smooth.png')
    plt.savefig(save_path + '_smooth.eps')


# Note: currently finds mean/median/min/max across episodes, then finds
# mean/median/min/max across JSONs to plot. Might not be equal to pooling all
# evaluation episodes together (across JSONs) and finding mean/median/min/max
# of that.
#
# Also, error_bar_stdev is calculated across the calculated mean/median/min/max
# of JSONs.
def plot_statistics(all_model_statistics, graphs_directory,
        json_mean_or_median, episode_mean_or_median, json_name=None,
        plotting_skip=0, skipped_checkpoints='skip', error_bar_stdev=0):
    # Assumes model filenames are of the form model.x.pth
    checkpoints = []
    all_returns = []
    all_successes = []
    skip_count = 0
    skip_all_returns = []
    skip_all_successes = []
    for checkpoint_name in all_model_statistics.keys():
        # Skip if no evaluation data for that json (only for single JSON plot)
        if json_name is not None and json_name not in all_model_statistics[checkpoint_name].keys():
            print("Trying to plot json " + json_name + " but no data for checkpoint " + checkpoint_name)
            continue

        num_jsons = len(all_model_statistics[checkpoint_name].keys())

        if num_jsons == 0:
            print("No JSONs but directory exists for " + checkpoint_name)
            continue

        if skip_count < plotting_skip and skipped_checkpoints == 'skip':
            skip_count += 1
            continue
        elif skip_count == plotting_skip and skipped_checkpoints == 'skip':
            skip_count = 0

        jsons_returns = []
        jsons_successes = []
        single_json_returns_stdev = 0.
        single_json_successes_stdev = 0.

        if json_name is not None:
            jsons_to_plot = [json_name]
        else:
            jsons_to_plot = all_model_statistics[checkpoint_name].keys()

        for json in jsons_to_plot:
            if jsons_name is not None:
                single_json_returns_stdev = np.stdev(all_model_statistics[checkpoint_name][json]['all_episode_returns'])
                single_json_successes_stdev = np.stdev(all_model_statistics[checkpoint_name][json]['all_episode_successes'])

            if episode_mean_or_median == 'mean':
                jsons_returns.append(np.mean(all_model_statistics[checkpoint_name][json]['all_episode_returns']))
                jsons_successes.append(np.mean(all_model_statistics[checkpoint_name][json]['all_episode_successes']))
            elif episode_mean_or_median == 'median':
                jsons_returns.append(np.median(all_model_statistics[checkpoint_name][json]['all_episode_returns']))
                jsons_successes.append(np.median(all_model_statistics[checkpoint_name][json]['all_episode_successes']))
            elif episode_mean_or_median == 'min':
                jsons_returns.append(np.min(all_model_statistics[checkpoint_name][json]['all_episode_returns']))
                jsons_successes.append(np.min(all_model_statistics[checkpoint_name][json]['all_episode_successes']))
            elif episode_mean_or_median == 'max':
                jsons_returns.append(np.max(all_model_statistics[checkpoint_name][json]['all_episode_returns']))
                jsons_successes.append(np.max(all_model_statistics[checkpoint_name][json]['all_episode_successes']))


        jsons_returns = np.array(jsons_returns)
        jsons_successes = np.array(jsons_successes)

        # If only plotting one JSON, use stdev over episodes. Otherwise, use
        # stdev over JSONs
        jsons_returns_stdev = single_json_returns_stdev if json_name is not None else np.std(jsons_returns)
        jsons_successes_stdev = single_json_successes_stdev if json_name is not None else np.std(jsons_successes)

        if skipped_checkpoints in ['mean', 'median', 'min', 'max']:

            if json_mean_or_median == 'mean':
                skip_all_returns.append(np.mean(jsons_returns), jsons_returns_stdev)
                skip_all_successes.append((np.mean(jsons_successes), jsons_successes_stdev))
            elif json_mean_or_median == 'median':
                skip_all_returns.append((np.median(jsons_returns), jsons_returns_stdev))
                skip_all_successes.append((np.median(jsons_successes), jsons_successes_stdev))
            elif json_mean_or_median == 'min':
                skip_all_returns.append((np.min(jsons_returns), jsons_returns_stdev))
                skip_all_successes.append((np.min(jsons_successes), jsons_successes_stdev))
            elif json_mean_or_median == 'max':
                skip_all_returns.append((np.max(jsons_returns), jsons_returns_stdev))
                skip_all_successes.append((np.max(jsons_successes), jsons_successes_stdev))

            if skip_count == plotting_skip - 1:
                skip_all_returns_means, skip_all_returns_stdevs = zip(*skip_all_returns)
                skip_all_successes_means, skip_all_successes_stdevs = zip(*skip_all_successes)

                if skipped_checkpoints == 'mean':
                    # Treat stdev of mean as mean of stdevs
                    all_returns.append((np.mean(np.array(skip_all_returns_means)),
                        np.mean(np.array(skip_all_returns_stdevs))))
                    all_successes.append((np.mean(np.array(skip_all_successes_means)),
                        np.mean(np.array(skip_all_successes_stdevs))))
                elif skipped_checkpoints == 'median':
                    # Use stdev of median value
                    # Sort by mean, then pick out stdev
                    sorted_skip_all_returns_means, sorted_skip_all_returns_stdevs = zip(
                        *sorted(skip_all_returns))
                    all_returns.append((
                        sorted_skip_all_returns_means[len(sorted_skip_all_returns_means) / 2],
                        sorted_skip_all_returns_stdevs[len(sorted_skip_all_returns_stdevs) / 2]))
                    all_successes.append(np.median(np.array(skip_all_successes)))
                elif skipped_checkpoints == 'min':
                    # Use stdev of min value
                    all_returns.append((np.min(np.array(skip_all_returns_mean)),
                        skip_all_returns_stdevs[np.argmin(np.array(skip_all_returns_mean))]))
                    all_successes.append((np.min(np.array(skip_all_successes_mean)),
                        skip_all_successes_stdevs[np.argmin(np.array(skip_all_successes_mean))]))
                elif skipped_checkpoints == 'max':
                    # Use stdev of max value
                    all_returns.append((np.max(np.array(skip_all_returns_mean)),
                        skip_all_returns_stdevs[np.argmax(np.array(skip_all_returns_mean))]))
                    all_successes.append((np.max(np.array(skip_all_successes_mean)),
                        skip_all_successes_stdevs[np.argmax(np.array(skip_all_successes_mean))]))

                checkpoints.append(int(checkpoint_name.split('.')[1]))
                skip_all_returns = []
                skip_all_successes = []

            skip_count = (skip_count + 1) % plotting_skip
            continue


        if json_mean_or_median == 'mean':
            all_returns.append((np.mean(jsons_returns), jsons_returns_stdev))
            all_successes.append((np.mean(jsons_successes), jsons_successes_stdev))
        elif json_mean_or_median == 'median':
            all_returns.append((np.median(jsons_returns), jsons_returns_stdev))
            all_successes.append((np.median(jsons_successes), jsons_successes_stdev))
        elif json_mean_or_median == 'min':
            all_returns.append((np.min(jsons_returns), jsons_returns_stdev))
            all_successes.append((np.min(jsons_successes), jsons_successes_stdev))
        elif json_mean_or_median == 'max':
            all_returns.append((np.max(jsons_returns), jsons_returns_stdev))
            all_successes.append((np.max(jsons_successes), jsons_successes_stdev))


        checkpoints.append(int(checkpoint_name.split('.')[1]))

    # sorted_returns_checkpoints, sorted_successes_checkpoints should be the same
    sorted_returns_checkpoints, sorted_all_returns =    \
        zip(*sorted(zip(checkpoints, all_returns)))
    sorted_all_returns_means, sorted_all_returns_stdevs =   \
        zip(*sorted_all_returns)

    sorted_successes_checkpoints, sorted_all_successes =    \
        zip(*sorted(zip(checkpoints, all_successes)))
    sorted_all_successes_means, sorted_all_successes_stdevs =   \
        zip(*sorted_all_successes)

    if plotting_skip > 0:
        save_path = os.path.join(graphs_directory, str(plotting_skip) +
            skipped_checkpoints + '_' + json_mean_or_median +
            episode_mean_or_median + str(error_bar_stdev) + '_evaluate')
    else:
        save_path = os.path.join(graphs_directory, json_mean_or_median +
            episode_mean_or_median + str(error_bar_stdev) + '_evaluate')

    graph(sorted_returns_checkpoints, sorted_all_returns_means,
        sorted_all_returns_stdevs, error_bar_stdev, 'returns',
        save_path + '_returns')
    graph(sorted_successes_checkpoints, sorted_all_successes_means,
        sorted_all_successes_stdevs, error_bar_stdev, 'successes',
        save_path + '_successes')

    # Print some statistics
    sorted_returns_checkpoints = np.array(sorted_returns_checkpoints)
    sorted_all_returns_means = np.array(sorted_all_returns_means)
    sorted_successes_checkpoints = np.array(sorted_successes_checkpoints)
    sorted_all_successes_means = np.array(sorted_all_successes_means)

    highest_return_index = np.argmax(sorted_all_returns_means)
    highest_return = sorted_all_returns_means[highest_return_index]
    highest_return_checkpoint = sorted_returns_checkpoints[highest_return_index]
    print('Highest average return : checkpoint {0}, return {1}'
            .format(highest_return_checkpoint, highest_return))
    highest_success_index = np.argmax(sorted_all_successes_means)
    highest_success = sorted_all_successes_means[highest_success_index]
    highest_success_checkpoint = sorted_successes_checkpoints[highest_success_index]
    print('Highest success rate : checkpoint {0}, successes {1}'
            .format(highest_success_checkpoint, highest_success))

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
            },
            # separate_directory is when we have directory -> ckpt eval dirs (no ckpts) -> JSON dirs
            'separate_directory' : {
                'name' : '--separate-directory',
                'type' : str,
                'metavar' : 'SD',
                'help' : 'If evaluations are in a separate directory, in separate directories'
            },
            'jsons_average' : {
                'name' : '--jsons-average',
                'type' : str,
                'metavar' : 'JA',
                'help' : '\'mean\', \'median\', \'min\', or \'max\' over JSONs'
            },
            'episodes_average' : {
                'name' : '--episodes-average',
                'type' : str,
                'metavar' : 'EA',
                'help' : '\'mean\', \'median\', \'min\', or \'max\' over episodes for a given JSON'
            },
            'plotting_skip' : {
                'name' : '--plotting-skip',
                'type' : int,
                'metavar' : 'PI',
                'help' : 'How many checkpoints to skip between plotted checkpoints'
            },
            'skipped_checkpoints' : {
                'name' : '--skipped-checkpoints',
                'type' : str,
                'metavar' : 'SC',
                'help' : 'Whether to "skip" or average via "mean" or "median" skipped checkpoints'
            },
            'error_bar_stdev' : {
                'name' : '--error-bar-stdev',
                'type' : int,
                'metavar' : 'EBS',
                'help' : 'How many standard deviations to include in the error bars'
            }
        },
        additional_default_args={
            'model_directory' : '',
            'evaluation_prefix' : '',
            'graphs_directory' : '',
            'separate_directory' : '',
            'jsons_average' : 'mean',
            'episodes_average' : 'mean',
            'plotting_skip' : 0,
            'skipped_checkpoints' : 'skip',
            'error_bar_stdev' : 0
        }
    )

    if not os.path.isdir(args['graphs_directory']):
        os.makedirs(args['graphs_directory'])

    if args['separate_directory'] != '':
        # directory -> checkpoints -> JSONS
        # In this case, model_directory is just the
        models_list = [f for f in os.listdir(args['separate_directory']) if '.pth' in f]
    else:
        # Directories go model -> checkpoints -> JSON -> evaluation_statistics.pth
        models_list = [f for f in os.listdir(args['model_directory']) if not os.path.isdir(os.path.join(args['model_directory'], f)) and '.pth' in f]

    # model_statistics is (checkpoint, (json, (all_episode_returns | all_episode_successes, np array))
    all_model_statistics = {}
    for model in models_list:
        current_model_statistics = {}

        if args['separate_directory'] != '':
            current_evaluation_directory = os.path.join(args['separate_directory'], model)
        else:
            current_evaluation_directory = os.path.join(args['model_directory'], args['evaluation_prefix'] + model)
        if os.path.isdir(current_evaluation_directory):
            for json_subdir in [f for f in os.listdir(current_evaluation_directory) if os.path.isdir(os.path.join(current_evaluation_directory, f))]:
                current_evaluation_statistics_path = os.path.join(current_evaluation_directory, json_subdir, 'evaluation_statistics.pth')

                # Makes the assumption that the json directory is [json filename]+junk
                json_name = json_subdir.split('.json')[0]
                if os.path.isfile(current_evaluation_statistics_path):
                    json_evaluation_statistics = torch.load(current_evaluation_statistics_path)
                    current_model_statistics[json_name] = json_evaluation_statistics

            all_model_statistics[model] = current_model_statistics
        else:
            print("No evaluation yet for model: " + model)

    plot_statistics(all_model_statistics, args['graphs_directory'],
        args['jsons_average'], args['episodes_average'],
        plotting_skip = args['plotting_skip'],
        skipped_checkpoints = args['skipped_checkpoints'],
        error_bar_stdev = args['error_bar_stdev'])

    # Plot results for each JSONs
    done_jsons = []
    for model in all_model_statistics.keys():
        for json in all_model_statistics[model].keys():
            if json not in done_jsons:
                json_graphs_directory = os.path.join(args['graphs_directory'], json)
                if not os.path.exists(json_graphs_directory):
                    os.mkdir(json_graphs_directory)
                # Note that the mean/median doesn't matter for JSONs since there's only one
                plot_statistics(all_model_statistics, json_graphs_directory,
                    args['jsons_average'], args['episodes_average'],
                    json_name = json,
                    plotting_skip = args['plotting_skip'],
                    skipped_checkpoints = args['skipped_checkpoints'],
                    error_bar_stdev = args['error_bar_stdev'])

                done_jsons.append(json)

