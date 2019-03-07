import os, datetime
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import re
import argparse
import inflect  # For converting integers to ordinals (first, second, etc)
p = inflect.engine()

from common.stat_utils import smooth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-directory-evaluation-prefix-labels',
        type=str,
        nargs='+',
        default='',
        metavar='MDEPL',
        help='Which directories to find model checkpoints in, the prefixes of each evaluation dataset\'s evaluation directories (not including last model.x.pth). Assumes that each model directory has a series of directories with this prefix, containing a directory for each JSON, containing evaluation_statistics.pth. Give as separate strings model-directory evaluation-prefix label for each desired line to be plot')
    parser.add_argument(
        '--graphs-directory',
        type=str,
        default='',
        metavar='GD',
        help='The directory in which to save the graphs')
    parser.add_argument(
        '--data-directory',
        type=str,
        default=None,
        metavar='DD',
        help='The directory in which to save the gathered evaluation data (default is None for no saving)')
    parser.add_argument(
        '--x-axis-factors',
        type=str,
        nargs='+',
        default=None,
        metavar='XAF',
        help='If provided, pairs of label, and factor to multiply x axis for that label. Note that labels should be unique if this option is used. ')
    # TODO(josh): add separate_directory functionality
    parser.add_argument(
        '--separate-directories',
        type=str,
        nargs='+',
        default='',
        metavar='SD',
        help='If evaluations are in a separate directory, in separate directories')
    parser.add_argument(
        '--average-to-use',
        type=str,
        default='mean',
        metavar='ATU',
        help='\'mean\', \'median\', \'min\', or \'max\' over results. Use two to separate average over JSONs and average over episodes (first is average over JSONs, second is average over episodes). Otherwise, averages over all episodes regardless of JSON')
    parser.add_argument(
        '--plotting-skip',
        type=int,
        default=0,
        metavar='PS',
        help='How many checkpoints to skip between plotted checkpoints')
    parser.add_argument(
        '--skipped-checkpoints',
        type=str,
        default='skip',
        metavar='SC',
        help='Whether to "skip" or average via "mean" or "median" skipped checkpoints')
    parser.add_argument(
        '--ci',
        type=int,
        default=95,
        metavar='CI',
        help='Size of confidence interval to show in error band (0 for no error band, 95 for 2 stdevs, \'sd\' for one standard deviation')
    parser.add_argument(
        '--smoothing-std',
        type=int,
        default=1,
        metavar='SS',
        help='std to pass to smoothing function (default: 1)')
    parser.add_argument(
        '--figure-width',
        type=float,
        default=6.4,
        metavar='FW',
        help='Width of generated figure')
    parser.add_argument(
        '--figure-height',
        type=float,
        default=4.8,
        metavar='FW',
        help='Height of generated figure')
    parser.add_argument('--successes-fixed-axis', dest='successes_fixed_axis',
            action='store_true')
    parser.add_argument('--successes-variable-axis',
            dest='successes_fixed_axis', action='store_false')
    parser.set_defaults(successes_fixed_axis=False)
    parser.add_argument(
        '--plot-until',
        type=int,
        default=None,
        metavar='PU',
        help='Last checkpoint number to plot (do not plot evaluations of later checkpoints)')

    return vars(parser.parse_args())

# Remember that df has columns model_directory, evaluation_prefix, checkpoint,
# json, return, and success
# TODO(josh): make plotting_skip do something
def plot_statistics(df, graphs_directory, average_to_use, json_name=None,
        plotting_skip=0, skipped_checkpoints='skip', ci=95, smoothing_std=1,
        successes_fixed_axis=False, plot_until=None):
    if average_to_use == 'mean':
        estimator = 'mean'
    elif average_to_use == 'median':
        estimator = 'median'
    elif average_to_use == 'min':
        estimator = 'min'
    elif average_to_use == 'max':
        estimator = 'max'
    else:   # TODO(josh): make separate JSON and episode estimators
        # First is over JSONs
        if average_to_use[:4] == 'mean':
            pass
        elif average_to_use[:6] == 'median':
            pass
        elif average_to_use[:3] == 'min':
            pass
        elif average_to_use[:3] == 'max':
            pass
        # Second is over episodes
        if average_to_use[-4:] == 'mean':
            pass
        elif average_to_use[-6:] == 'median':
            pass
        elif average_to_use[-3:] == 'min':
            pass
        elif average_to_use[-3:] == 'max':
            pass

    # TODO(josh): automatically make json_name graphs instead of as an argument?
    if json_name is not None:
        # Filter out all data that doesn't have the json_name
        df = df[df['json'] == json_name]

    # Remove all data points with checkpoint greater than plot_until
    if plot_until is not None:
        df = df[df['checkpoint'] < plot_until]

    # Prepare statistics string
    out_string = ''
    if json_name is not None:
        out_string = 'Single JSON statistics: {0} \n'.format(json_name)

    # Since smoothing is only defined over one dimension, we should find the
    # mean/median/min/max of return and success at each checkpoint for each
    # label over all jsons, and append that to a new data frame
    #all_returns_smooth = smooth(np.array(sorted_all_returns_means),
    #    np.array(sorted_checkpoints))
    data_smooth = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_df = label_df.loc[:, label_df.columns != 'label']
        current_data = []
        for checkpoint in label_df['checkpoint'].unique():
            checkpoint_df = label_df[label_df['checkpoint'] == checkpoint]
            checkpoint_df = checkpoint_df.loc[:, checkpoint_df.columns
                != 'checkpoint']
            no_json_df = checkpoint_df.loc[:, checkpoint_df.columns != 'json']
            avg_rt, avg_sc = no_json_df.agg(estimator)
            current_data.append((checkpoint, avg_rt, avg_sc))
        # Sort and do smoothing
        sorted_checkpoints, sorted_avg_rt, sorted_avg_sc = zip(*sorted(current_data))
        sorted_checkpoints = np.array(sorted_checkpoints)
        sorted_avg_rt = np.array(sorted_avg_rt)
        sorted_avg_sc = np.array(sorted_avg_sc)
        sorted_avg_rt_smooth = smooth(np.array(sorted_avg_rt),
            np.array(range(len(sorted_checkpoints))), std=smoothing_std)
        sorted_avg_sc_smooth = smooth(np.array(sorted_avg_sc),
            np.array(range(len(sorted_checkpoints))), std=smoothing_std)

        #wanted_index = list(sorted_checkpoints).index(701130)
        #print('701130 successes: ' + str(sorted_avg_sc[wanted_index]))

        for i in range(len(sorted_checkpoints)):
            data_smooth.append([label, sorted_checkpoints[i],
                sorted_avg_rt_smooth[i],
                sorted_avg_sc_smooth[i]])

        # Find best_n best scoring checkpoints for this label and print
        best_n = min(10, len(sorted_avg_rt))
        highest_rt_indexes = np.argpartition(sorted_avg_rt, -best_n)[-best_n:]
        highest_rt_indexes = highest_rt_indexes[np.argsort(
            sorted_avg_rt[highest_rt_indexes])[::-1]]
        highest_sc_indexes = np.argpartition(sorted_avg_sc, -best_n)[-best_n:]
        highest_sc_indexes = highest_sc_indexes[np.argsort(
            sorted_avg_sc[highest_sc_indexes])[::-1]]

        for i in range(best_n):
            # Sometimes will use this functionality for finding the best
            # checkpoint among some very incomplete evaluations (i.e. test), so
            # might not always have best_n checkpoints
            if i > len(highest_rt_indexes):
                break
            out_string += (p.ordinal(i + 1) + ' highest average return for '
                    + label + ': checkpoint {0}, return {1} \n'
                    .format(sorted_checkpoints[highest_rt_indexes[i]],
                        sorted_avg_rt[highest_rt_indexes[i]]))
        out_string += '\n'
        for i in range(best_n):
            if i > len(highest_sc_indexes):
                break
            out_string += (p.ordinal(i + 1) + ' highest average successes for '
                    + label + ': checkpoint {0}, successes {1} \n'
                    .format(sorted_checkpoints[highest_sc_indexes[i]],
                        sorted_avg_sc[highest_sc_indexes[i]]))
        out_string += '\n'

    df_smooth = pd.DataFrame(columns=['label',
        'checkpoint', 'return', 'success'], data=data_smooth)


    fig_dims = (args['figure_width'], args['figure_height'])
    # Print and save statistics
    print(out_string)

    labels_str = '-'.join(df['label'].unique().tolist())
    with open(os.path.join(graphs_directory, labels_str + average_to_use
        + str(ci) + '_evaluate_') + 'stats.txt', 'w') as outfile:
        outfile.write(out_string)

    for return_or_success in ['return', 'success']:
        for smoothing in [True, False]:
            if plotting_skip > 0:
                save_path = os.path.join(graphs_directory, labels_str
                    + str(plotting_skip) + skipped_checkpoints + '_'
                    + average_to_use + str(ci) + '_evaluate_')
            else:
                save_path = os.path.join(graphs_directory, labels_str
                    + average_to_use + str(ci) + '_evaluate_')

            plt.clf()
            fig, ax = plt.subplots(figsize=fig_dims)

            sns.lineplot(ax=ax, x='checkpoint', y=return_or_success,
                hue='label', estimator=estimator, ci=ci,
                data=df_smooth if smoothing else df)
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0))
            # Remove legend title
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])
            # Draw legend outside of graph
            #lgd = plt.legend(loc='upper left')
            plt.xlabel('Gradient steps')  # Model checkpoints
            if return_or_success == 'return':
                #plt.title('Directory Evaluation Returns')
                plt.ylabel('Reward') # Average reward per episode
                save_path += 'returns'
            elif return_or_success == 'success':
                #plt.title('Directory Evaluation Successes')
                if successes_fixed_axis:
                    ax.set_ylim(-0.01, 1.01)
                plt.ylabel('Success rate') # Average success per episode
                save_path += 'successes'

            if smoothing:
                save_path += '_smooth'

            save_path += datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")

            # Remove bbox_extra_artists if not drawing legend outside of box
            plt.savefig(save_path + '.png', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
            plt.savefig(save_path + '.eps', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')


# In a single graph, will graph tuple of (model-directory, evaluation-prefix)
# TODO(josh): group evaluations into better, nested directories, Should be
# model { checkpoints (pth), evaluation datasets { checkpoints { JSONs
if __name__=='__main__':
    args = parse_args()

    if not os.path.isdir(args['graphs_directory']):
        os.makedirs(args['graphs_directory'])

    # Entries in data are ordered in: model_directory, evaluation_prefix,
    # checkpoint, json, single episode return
    # TODO(josh): see if this is fast or if we should construct dataframe from
    # dict of dicts, etc
    data = []
    model_directory_evaluation_prefix_labels = []
    for i in range(0, len(args['model_directory_evaluation_prefix_labels']), 3):
        md = args['model_directory_evaluation_prefix_labels'][i]
        ep = args['model_directory_evaluation_prefix_labels'][i + 1]
        label = args['model_directory_evaluation_prefix_labels'][i + 2]
        model_directory_evaluation_prefix_labels.append((md, ep, label))
    for model_directory, evaluation_prefix, label in model_directory_evaluation_prefix_labels:
        # Directories go model -> checkpoints -> JSON -> evaluation_statistics.pth
        checkpoints_list = [f for f in os.listdir(model_directory) if not os.path.isdir(os.path.join(model_directory, f)) and '.pth' in f]
        for checkpoint in checkpoints_list:
            # Assumes checkpoints are of the form 'model.x.pth'
            checkpoint_number = int(checkpoint.split('.')[1])
            current_evaluation_directory = os.path.join(model_directory,
                evaluation_prefix + checkpoint)

            # model_statistics is (checkpoint, (json, (all_episode_returns | all_episode_successes, np array))
            if os.path.isdir(current_evaluation_directory):
                # TODO(josh): make this scan over JSONs in the dataset directory (an additional argument), not the evaluations directory
                for json_subdir in [f for f in os.listdir(current_evaluation_directory) if os.path.isdir(os.path.join(current_evaluation_directory, f))]:
                    current_evaluation_statistics_path = os.path.join(current_evaluation_directory, json_subdir, 'evaluation_statistics.pth')

                    # Makes the assumption that the json directory is [json filename]+junk
                    json_name = json_subdir.split('.json')[0]
                    if os.path.isfile(current_evaluation_statistics_path):
                        json_evaluation_statistics = torch.load(current_evaluation_statistics_path)
                        for (rt, sc) in zip(
                            json_evaluation_statistics['all_episode_returns'],
                            json_evaluation_statistics['all_episode_successes']):
                            data.append([label, checkpoint_number, json_name, rt, sc])
                    else:
                        print("Directory exists but no evaluation_statistics at " + current_evaluation_statistics_path)
            else:
                print("No evaluation yet for model: " + checkpoint + " at "
                    + current_evaluation_directory)

    df = pd.DataFrame(columns=['label',
        'checkpoint', 'json', 'return', 'success'], data=data)
    if args['x_axis_factors'] is not None:
        for i in range(0, len(args['x_axis_factors']), 2):
            label = args['x_axis_factors'][i]
            factor = int(args['x_axis_factors'][i + 1])
            is_label = df['label'] == label
            # Multiplies entries in rows with is_label in column 'checkpoint'
            # by factor
            df.loc[is_label, 'checkpoint'] *= factor

    plot_statistics(df, args['graphs_directory'], args['average_to_use'],
        plotting_skip = args['plotting_skip'],
        skipped_checkpoints = args['skipped_checkpoints'],
        ci = args['ci'], smoothing_std = args['smoothing_std'],
        successes_fixed_axis = args['successes_fixed_axis'],
        plot_until = args['plot_until'])

    labels_str = '-'.join(df['label'].unique().tolist())
    if args['data_directory'] is not None:
        np.save(os.path.join(args['data_directory'], labels_str
            + '-evaluations.npy'), df.to_numpy())

    # TODO(josh): make single JSON plots easier with the magic of pandas dataframes
    '''
    # Plot results for each JSONs
    done_jsons = []
    for model in data.keys():
        for checkpoint in data[model].keys():
            for evaluation_prefix in data[model][checkpoint].keys():
                for json in data[model][checkpoint][evaluation_prefix].keys():
                    if json not in done_jsons:
                        json_graphs_directory = os.path.join(
                            args['graphs_directory'], json)
                        if not os.path.exists(json_graphs_directory):
                            os.mkdir(json_graphs_directory)
                        # Note that the mean/median doesn't matter for JSONs
                        # since there's only one
                        plot_statistics(data,
                            json_graphs_directory,
                            args['average_to_use'],
                            json_name = json,
                            plotting_skip = args['plotting_skip'],
                            skipped_checkpoints = args['skipped_checkpoints'],
                            ci = args['ci'])

                        done_jsons.append(json)
    '''
