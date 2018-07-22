from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import importlib

import torch
from torch.autograd import Variable

from common.environment import create_env
from common.utils import setup_logger

from a3g.player_util import Agent

import os
import time
import copy
import logging
import gym

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def test(args, shared_model, optimizer, all_scores):
    # Shortcut to save directory
    save_dir = args['save_directory']+'/'
    run_name = os.path.basename(args['save_directory'].strip('/'))

    ptitle('Test Agent')
    gpu_id = args['gpu_ids'][-1]
    log = {}
    setup_logger('info.log', r'{0}/info.log'.format(save_dir))
    log['info.log'] = logging.getLogger('info.log')
    for k in args.keys():
        log['info.log'].info('{0}: {1}'.format(k, args[k]))

    torch.manual_seed(args['seed'])
    if gpu_id >= 0:
        torch.cuda.manual_seed(args['seed'])
    env = create_env(args['env'], args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id

    AC = importlib.import_module(args['model_name'])
    player.model = AC.ActorCritic(
        env.observation_space, env.action_space, args['stack_frames'], args)
    best_model = AC.ActorCritic(
        env.observation_space, env.action_space, args['stack_frames'], args)
    best_optimizer_state_dict = None
    if optimizer is not None:
        best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    player.state, player.info = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    player.model.eval()

    episode_count = len(all_scores)
    max_score = np.max(all_scores) if len(all_scores) > 0 else 0
    while True:
        if player.done:
            episode_count += 1
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())

        player.action_test()
        reward_sum += player.reward

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['info.log'].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            # Plot scores every 5 episodes
            all_scores.append(reward_sum)
            if (episode_count%5 == 0):
                plt.clf()
                plt.plot(range(len(all_scores)), all_scores)
                plt.title('Test Episode Returns')
                plt.xlabel('Test Episode')
                plt.ylabel('Return')
                plt.savefig('{0}/test_episode_returns.png'.format(save_dir))

            if reward_sum >= max_score:
                max_score = reward_sum
                best_model.load_state_dict(player.model.state_dict())

                optimizer_state_dict = None
                if optimizer is not None:
                    optimizer_state_dict = optimizer.state_dict()
                    best_optimizer_state_dict = optimizer.state_dict()

                torch.save({
                    'args' : args,
                    'episode_count' : episode_count,
                    'state_dict' : player.model.state_dict(),
                    'best_state_dict' : best_model.state_dict(),
                    'optimizer' : optimizer_state_dict,
                    'best_optimizer' : best_optimizer_state_dict,
                    'all_scores' : all_scores,
                }, save_dir+'model.pth')

            # Save every 25 episodes or max episode
            if (episode_count%25 == 0):
                optimizer_state_dict = None
                if optimizer is not None:
                    optimizer_state_dict = optimizer.state_dict()

                torch.save({
                    'args' : args,
                    'episode_count' : episode_count,
                    'state_dict' : player.model.state_dict(),
                    'best_state_dict' : best_model.state_dict(),
                    'optimizer' : optimizer_state_dict,
                    'best_optimizer' : best_optimizer_state_dict,
                    'all_scores' : all_scores,
                }, save_dir+'model.pth')

            reward_sum = 0
            player.eps_len = 0
            state, player.info = player.env.reset()
            time.sleep(60)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
