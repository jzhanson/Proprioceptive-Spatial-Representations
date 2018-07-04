from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import importlib

import torch
import torch.optim as optim
from torch.autograd import Variable


from imitate.environment import create_env
from imitate.utils import ensure_shared_grads
from imitate.player_util import Agent

import gym


def train(rank, args, shared_model, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = create_env(args.env, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, None, env, args, None)
    player.gpu_id = gpu_id
    AC = importlib.import_module(args.model_name)
    player.model = AC.ActorCritic(
        env.observation_space, env.action_space, args.stack_frames)
    EXP = importlib.import_module(args.expert_model_name)
    player.expert = EXP.ActorCritic(
        env.observation_space, env.action_space, args.stack_frames)
    player.expert.load_state_dict(shared_expert.state_dict())

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()

    step_count = 0
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.memory = player.model.initialize_memory()
                    player.expert_memory = player.expert.initialize_memory()
            else:
                player.memory = player.model.initialize_memory()
                player.expert_memory = player.expert.initialize_memory()
        else:
            player.memory = player.model.reinitialize_memory(player.memory)
            player.expert_memory = player.expert.reinitialize_memory(player.expert_memory)
            
        for step in range(args.num_steps):

            player.action_train()

            if player.done:
                break

        if player.done:
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        # Imitation + Entropy loss
        policy_loss = 0
        for i in reversed(range(len(player.rewards))):
            policy_loss = policy_loss - \
                          (player.ces[i].sum()) - \
                          (0.01 * player.entropies[i].sum())

        player.model.zero_grad()
        policy_loss.backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

        step_count += 1
        if (rank == 0) and (step_count%500) == 0:
            print('Model weight/gradient L-inf norm:')
            def _linf_norm(x):
                return str(torch.max(torch.abs(x))[0].data.item())
            for pname, param in player.model.named_parameters():
                pgradnorm = str(0.)
                if param.grad is not None:
                    pgradnorm = _linf_norm(param.grad)
                    print('\t'+pname+' '+_linf_norm(param)+'/'+pgradnorm)
