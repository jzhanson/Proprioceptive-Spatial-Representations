import os
from evaluate import evaluate
import copy
import numpy as np
import torch
import torch.multiprocessing as mp
from args import parse_args
import time

# For a single model, runs an evaluation on all JSONs in a directory
def directory_evaluate(args):
    start_time = time.time()
    all_evaluation_statistics = {}

    files_list = [f for f in os.listdir(args['json_directory']) if '.json' in f]

    # Check whether evaluations have been completed on some JSONs
    output_dir = os.path.join(os.path.dirname(args['load_file']),
        args['output_directory'])
    all_statistics_output_path = os.path.join(output_dir,
                              'JSONWalker-'+(args['json_directory'].replace('/',
                                  '-'))+'-evaluation-statistics-evalep{}.pth'
                              .format(args['num_episodes']))

    if os.path.isdir(output_dir):
        done_jsons_directories = [f for f in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, f))
                and os.path.isfile(os.path.join(output_dir, f,
                    'evaluation_statistics.pth'))]
        # Makes the assumption that json directory is [json filename] + junk
        done_jsons = [f.split('.json')[0] + '.json' for f in done_jsons_directories]
        files_list = [f for f in files_list if f not in done_jsons]
        # If no files left, continue
        if len(files_list) == 0:
            return
        # TODO(josh): make this read all subdir JSON evaluation statistics and
        # repair all_evaluation_statistics or make a separate script to do that
        if os.path.isfile(all_statistics_output_path):
            all_evaluation_statistics = torch.load(all_statistics_output_path)

    for f in files_list:
        new_args = copy.deepcopy(args)
        new_args['env'] = 'JSONWalker-' + args['json_directory'] + '/' + f

        evaluation_statistics = evaluate(new_args)
        all_evaluation_statistics[new_args['env']] = evaluation_statistics

    # Save all evaluation statistics at the end of all evaluations
    # Note: saving all statistics after each JSON evaluate is likely to overload
    # the filesystem's write capacity
    torch.save({
        'all_evaluation_statistics' : all_evaluation_statistics,
    }, all_statistics_output_path)

    end_time = time.time()
    total_episodes = len(files_list) * args['num_episodes']
    print('directory evaluate total time for %d episodes: %d' % (total_episodes,
        end_time - start_time))
    print('directory evaluate overall seconds per episode: %f' %
        ((end_time - start_time) / total_episodes))
    #return all_evaluation_statistics


# Writes evaluations to the same directory as the models
if __name__=='__main__':
    args = parse_args(
        additional_parser_args={
            'num_episodes' : {
                'name' : '--num-episodes',
                'type' : int,
                'metavar' : 'NE',
                'help' : 'how many epiosdes in evaluation (default: 100)'
            },
            'json_directory' : {
                'name' : '--json-directory',
                'type' : str,
                'metavar' : 'JD',
                'help' : 'Directory with JSONS to run evaluate on'
            },
            'evaluation_prefix' : {
                'name' : '--evaluation-prefix',
                'type' : str,
                'metavar' : 'EP',
                'help' : 'Prefix for created evaluation directories'
            },
            'model_directory' : {
                'name' : '--model-directory',
                'type' : str,
                'metavar' : 'MD',
                'help' : 'Directory to load model files from (exclusive of model-path)'
            },
            'checkpoint_names' : {
                'name' : '--checkpoint-names',
                'type' : str,
                'metavar' : 'CN',
                'help' : 'Comma-separated checkpoint names of checkpoints inside model-directory to evaluate (will only evaluate these checkpoints)'
            },
            'num_processes' : {
                'name' : '--num-processes',
                'type' : int,
                'metavar' : 'NT',
                'help' : 'How many processes to run at a time'
            },
            'render_video' : {
                'name' : '--render-video',
                'type' : bool,
                'metavar' : 'RV',
                'help' : 'Whether or not to save videos'
            },
        },
        additional_default_args={
            'num_episodes' : 100,
            'json_directory' : '',
            'evaluation_prefix' : '',
            'model_directory' : None,
            'checkpoint_names' : None,
            'num_processes' : 10,
            'render_video' : False
        }
    )

    # TODO(josh): add all_model_statistics to saving
    all_model_statistics = []

    processes = []
    # Call directory_evaluate for every model and check whether the evaluations
    # were completed inside directory_evaluate
    models_list = []
    if args['checkpoint_names'] is not None:
        models_list = args['checkpoint_names'].split(',')
    else:
        models_list = [f for f in os.listdir(args['model_directory']) if '.pth' in f
                and not os.path.isdir(os.path.join(args['model_directory'], f))]
    while len(models_list) > 0:
        if len(processes) < args['num_processes']:
            current_model = models_list.pop()
            print(current_model)
            current_args = copy.deepcopy(args)
            current_args['load_file'] = os.path.join(args['model_directory'], current_model)
            # Note that output_directory will be subdirectory of model_directory
            current_args['output_directory'] = args['evaluation_prefix']   \
                + current_model
            # TODO(josh): have to update all_model_statistics from
            # directory_evaluate since we can't return things from mp.Process
            p = mp.Process(target=directory_evaluate, args=(current_args,))
            p.start()
            processes.append(p)
            time.sleep(0.1)
            # TODO(josh): appending won't work since no guarantee will be read in order
            #all_model_statistics.append(current_model_statistics)
        # Try to join each process, and if successful, continue
        for i in range(len(processes)):
            time.sleep(0.1)
            processes[i].join(1.0)
            if processes[i].exitcode is not None:
                print("process " + str(i) + " exitcode: " + str(processes[i].exitcode))
                processes.pop(i)
                break
