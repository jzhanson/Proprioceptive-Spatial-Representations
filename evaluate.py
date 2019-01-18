from __future__ import division
import os
import datetime
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import importlib

import torch
from torch.autograd import Variable

import numpy as np
import numpy.random as npr

from common.environment import create_env
from common.utils import setup_logger

from a3g.player_util import Agent

from args import parse_args

import gym
import logging

import time

def evaluate(args):
    start_time = time.time()
    torch.set_default_tensor_type('torch.FloatTensor')

    start_loading = time.time()
    pthfile = torch.load(args['load_file'], map_location=lambda storage, loc: storage.cpu())
    end_loading = time.time()
    print('loading time: %d' % (end_loading - start_loading))

    # Create the output directory
    output_dir = os.path.join(os.path.dirname(args['load_file']), args['output_directory'], os.path.split(args['env'])[1]+'evaluation-'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))
    try:
        os.makedirs(output_dir)
    except OSError:
        if not os.path.isdir(output_dir):
            raise
    print('saving to: '+output_dir+'/')

    log = {}
    setup_logger('test.log', r'{0}/test.log'.format(output_dir))
    log['test.log'] = logging.getLogger('test.log')

    gpu_id = args['gpu_ids'][-1]

    torch.manual_seed(args['seed'])
    npr.seed(args['seed']+1)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args['seed'])

    for k in args.keys():
        log['test.log'].info('{0}: {1}'.format(k, args[k]))

    env = create_env(args['env'], args)
    player = Agent(None, env, args, None)

    # Wrap the environment so that it saves a video
    if args['render_video']:
        player.env = gym.wrappers.Monitor(player.env, output_dir, force=True)

    AC = importlib.import_module(args['model_name'])
    player.model = AC.ActorCritic(
        env.observation_space, env.action_space, args['stack_frames'], args)

    player.gpu_id = gpu_id
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()

    if args['load_best']:
        player.model.load_state_dict(pthfile['best_state_dict'])
    else:
        player.model.load_state_dict(pthfile['state_dict'])
    player.model.eval()

    end_setup = time.time()
    print('evaluate setup time: %d' % (end_setup - start_time))

    # Keep track of returns
    all_episode_returns = []
    for i_episode in range(args['num_episodes']):
        player.state, player.info = player.env.reset()
        player.state = torch.from_numpy(player.state).float()
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()
        player.eps_len = 0
        reward_sum = 0
        episode_step = 0
        while True:
            player.action_test(episode_step)
            reward_sum += player.reward
            episode_step += 1

            if player.done:
                all_episode_returns.append(reward_sum)
                #num_tests += 1
                #reward_total_sum += reward_sum
                #reward_mean = reward_total_sum / num_tests
                log['test.log'].info(
                    "Episode_length, {0}, reward_sum, {1}".format(player.eps_len, reward_sum))
                break
    end_episodes = time.time()
    print('single evaluate time for %d episodes: %d' %
        (args['num_episodes'], end_episodes - end_setup))
    print('single evaluate seconds per episode: %d' %
        ((end_episodes - end_setup) / args['num_episodes']))
    all_episode_returns = np.array(all_episode_returns)
    all_episode_successes = np.array(all_episode_returns > 300., dtype=np.float32)

    evaluation_statistics = {
        'Mean Return': np.mean(all_episode_returns),
        'Std Return': np.std(all_episode_returns),
        'Min Return': np.min(all_episode_returns),
        'Max Return' : np.max(all_episode_returns),

        'Mean Success': np.mean(all_episode_successes),
        'Number Successes': np.sum(all_episode_successes),
        'Number Total': args['num_episodes'],
        'Std Success': np.std(all_episode_successes),
        'Min Success': np.min(all_episode_successes),
        'Max Success' : np.max(all_episode_successes),

        'all_episode_returns': all_episode_returns,
        'all_episode_successes': all_episode_successes,
    }

    # Save raw data to a file
    start_saving = time.time()
    torch.save({
        'all_episode_returns' : all_episode_returns,
        'all_episode_successes' : all_episode_successes,
    }, os.path.join(output_dir, 'evaluation_statistics.pth'))
    end_saving = time.time()
    print('time spent saving single evaluation: %d' % (end_saving - start_saving))

    print('Average Episodic Return: \n\tmean: {0}\n\tstd: {1}\n\tmin: {2}\n\tmax: {3}'.format(
        np.mean(all_episode_returns), np.std(all_episode_returns),
        np.min(all_episode_returns), np.max(all_episode_returns)))
    print('Average Episodic Success: \n\tmean: {0} ({1}/{2})\n\tstd: {3}\n\tmin: {4}\n\tmax: {5}'.format(
        np.mean(all_episode_successes), np.sum(all_episode_successes), args['num_episodes'],
        np.std(all_episode_successes), np.min(all_episode_successes), np.max(all_episode_successes)))

    # Shut down logging system and close open file handles
    logging.shutdown()

    end_time = time.time()
    print('single evaluate total time for %d episodes: %d' % (total_episodes,
        end_time - start_time))
    print('single evaluate overall seconds per episode: %f' %
        ((end_time - start_time) / total_episodes))
    return evaluation_statistics

if __name__=='__main__':
    evaluate(parse_args(
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
            'output_directory' : {
                'name' : '--output-directory',
                'metavar' : 'OD',
                'help' : 'Directory to write output to'
            }
        },
        additional_default_args={
            'num_episodes' : 100,
            'render_video' : False,
            'output_directory' : ''
        }
    ))
