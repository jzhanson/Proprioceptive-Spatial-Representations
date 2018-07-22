from __future__ import division
import os
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


def evaluate(args):
    torch.set_default_tensor_type('torch.FloatTensor')

    pthfile = torch.load(args['load_file'], map_location=lambda storage, loc: storage.cpu())

    save_dir = args['save_directory']+'/'

    log = {}
    setup_logger('test.log', r'{0}/test.log'.format(save_dir))
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
        while True:
            player.action_test()
            reward_sum += player.reward

            if player.done:
                all_episode_returns.append(reward_sum)
                #num_tests += 1
                #reward_total_sum += reward_sum
                #reward_mean = reward_total_sum / num_tests
                log['test.log'].info(
                    "Episode_length, {0}, reward_sum, {1}".format(player.eps_len, reward_sum))
                break
    all_episode_returns = np.array(all_episode_returns)

    print('Average Episodic Return: \n\tmean: {0}\n\tstd: {1}\n\tmin: {2}\n\tmax: {3}'.format(
        np.mean(all_episode_returns), np.std(all_episode_returns),
        np.min(all_episode_returns), np.max(all_episode_returns)))

    all_episode_successes = np.array(all_episode_returns > 300., dtype=np.float32)
    print('Average Episodic Success: \n\tmean: {0} ({1}/{2})\n\tstd: {3}\n\tmin: {4}\n\tmax: {5}'.format(
        np.mean(all_episode_successes), np.sum(all_episode_successes), args['num_episodes'],
        np.std(all_episode_successes), np.min(all_episode_successes), np.max(all_episode_successes)))

if __name__=='__main__':
    evaluate(parse_args(
        additional_parser_args={
            'num_episodes' : {
                'name' : '--num-episodes',
                'type' : int,
                'metavar' : 'NE',
                'help' : 'how many epiosdes in evaluation (default: 100)'
            }
        },
        additional_default_args={
            'num_episodes' : 100
        }
    ))
