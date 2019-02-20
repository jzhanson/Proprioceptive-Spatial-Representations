from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import importlib

import numpy        as np
import numpy.random as npr

import torch
import torch.multiprocessing as mp

from common.environment import create_env
from common.shared_optim import SharedRMSprop, SharedAdam

from a3g.train import train
from a3g.test import test

from args import parse_args

import time

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

def main(args):
    torch.manual_seed(args['seed'])
    npr.seed(args['seed']+1)

    # Create the save directory
    try:
        os.makedirs(args['save_directory'])
    except OSError:
        if not os.path.isdir(args['save_directory']):
            raise
    print('saving to: '+args['save_directory']+'/')

    if args['gpu_ids'] == -1:
        args['gpu_ids'] = [-1]
    else:
        torch.cuda.manual_seed(args['seed'])
        mp.set_start_method('spawn')
    env = create_env(args['env'], args)

    # Create model
    AC = importlib.import_module(args['model_name'])
    shared_model = AC.ActorCritic(
        env.observation_space, env.action_space, args['stack_frames'], args)

    shared_model.share_memory()

    if args['shared_optimizer']:
        if args['optimizer'] == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args['lr'])
        if args['optimizer'] == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args['lr'], amsgrad=args['amsgrad'])
        optimizer.share_memory()
    else:
        optimizer = None

    # Keep track of all steps taken in each thread
    all_step_counters = [mp.Value('i', 0) for i in range(args['workers'])]
    global_step_counter = mp.Value('i', 0)


    # Keep track of stats if we want to load from a checkpoint
    all_scores = []
    all_global_steps = []
    if args['load_file'] != '':
        print('Loading model from: {0}'.format(args['load_file']))
        pthfile = torch.load('{0}'.format(args['load_file']), map_location=lambda storage, loc: storage.cpu())
        if args['load_best']:
            shared_model.load_state_dict(pthfile['best_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(pthfile['best_optimizer'])
        else:
            shared_model.load_state_dict(pthfile['state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(pthfile['optimizer'])
            all_scores = pthfile['all_scores']
            all_global_steps = pthfile['all_global_steps']

    # Only test process will write to this to avoid each thread waiting every
    # gradient step to update. Threads will read from global_step_counter to
    # know when to terminate if args['test_until'] is used
    if len(all_global_steps) > 0:
        # This increment doesn't have to be atomic
        with global_step_counter.get_lock():
            global_step_counter.value = all_global_steps[-1]

    processes = []

    p = mp.Process(target=test, args=(args, shared_model, optimizer,
                                      all_scores, all_global_steps,
                                      all_step_counters, global_step_counter))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args['workers']):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer, all_step_counters[rank],
            global_step_counter))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__=='__main__':
    main(parse_args())
