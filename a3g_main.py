from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import importlib

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
    if args['load']:
        print('Loading model from: {0}{1}.dat'.format(
            args['load_model_dir'], args['env']))
        saved_state = torch.load('{0}{1}.dat'.format(
            args['load_model_dir'], args['env']), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
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

    processes = []

    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args['workers']):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__=='__main__':
    main(parse_args())
