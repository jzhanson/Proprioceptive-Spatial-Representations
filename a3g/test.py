from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import importlib

import torch
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from common.environment import create_env
from common.utils import setup_logger
from common.stat_utils import smooth

from a3g.player_util import Agent

import os
import time
import copy
import logging
import gym

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def test(args, shared_model, optimizer, all_scores, all_global_steps,
        all_step_counters, global_step_counter):
    # Shortcut to save directory
    save_dir = args['save_directory']+'/'
    run_name = os.path.basename(args['save_directory'].strip('/'))

    gpu_id = args['gpu_ids'][-1]
    if args['experiment_id'] == '':
        ptitle('Test Agent')
    else:
        ptitle('EXPID: {} Test Agent'.format(args['experiment_id']))
    log = {}
    setup_logger('train.log', r'{0}/train.log'.format(save_dir))
    log['train.log'] = logging.getLogger('train.log')
    for k in args.keys():
        log['train.log'].info('{0}: {1}'.format(k, args[k]))

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

    # Runs from resumed value (start_global_step is constant)
    start_global_step = global_step_counter
    # Runs from 0, regardless of resumed value
    global_step = 0

    episode_step = 0
    episode_count = len(all_scores)
    max_score = np.max(all_scores) if len(all_scores) > 0 else 0

    # Set up tensorboardx
    writer = SummaryWriter()

    while True:
        if player.done:
            # Only run test episode every ~N updates
            new_global_step = global_step
            while (new_global_step - global_step) < args['test_every_n_steps']:
                # Get number of gradient steps asynchronously
                new_global_step = 0
                for i in range(len(all_step_counters)):
                    new_global_step += all_step_counters[i].value
                #print(new_global_step)
                time.sleep(2)
            global_step = new_global_step
            global_step_counter = start_global_step + global_step

            episode_count += 1
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())

        # player.action_test(episode_step)
        player.action_test()
        reward_sum += player.reward
        episode_step += 1

        if player.done:
            episode_step = 0
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['train.log'].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))


            all_global_steps.append(global_step_counter)
            all_scores.append(reward_sum)

            x = np.array(all_global_steps) #range(len(all_scores)))
            y = np.array(all_scores)

            # Raw plot
            plt.clf()
            plt.plot(x, y)
            plt.title('Test Episode Returns')
            plt.xlabel('Global gradient step')
            plt.ylabel('Test Episode Return')
            plt.savefig('{0}/test_episode_returns.png'.format(save_dir))
            plt.savefig('{0}/test_episode_returns.eps'.format(save_dir))

            # Smoothed version
            plt.clf()
            y_smooth = smooth(y, x)
            plt.plot(x, y_smooth, 'k', color='#CC4F1B')
            plt.title('Test Episode Returns')
            plt.xlabel('Global gradient step')
            plt.ylabel('Test Episode Return')
            plt.savefig('{0}/test_episode_returns_smooth.png'.format(save_dir))
            plt.savefig('{0}/test_episode_returns_smooth.eps'.format(save_dir))

            model_path = save_dir+'/model'
            if args['save_intermediate']:
                model_path = model_path+'.'+str(global_step_counter) #episode_count)
            model_path = model_path+".pth"

            # Is this the best model so far?
            if reward_sum >= max_score:
                max_score = reward_sum
                best_model.load_state_dict(player.model.state_dict())
                if optimizer is not None:
                    best_optimizer_state_dict = optimizer.state_dict()

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
                'all_global_steps' : all_global_steps,
            }, model_path)

            writer.add_scalar('max_score', max_score, global_step_counter)
            writer.add_scalar('reward_sum', reward_sum, global_step_counter)
            writer.add_scalar('reward_mean', reward_mean, global_step_counter)
            writer.add_scalar('player_eps_len', player.eps_len, global_step_counter)

            reward_sum = 0
            player.eps_len = 0
            state, player.info = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

            # Terminate testing after at least train_until steps
            if args['train_until'] is not None \
                and global_step_counter > args['train_until']:
                break

