from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from imitate.utils import normal  # , pi


class Agent(object):
    def __init__(self, model, expert, env, args, state):
        self.model = model
        self.expert = expert
        self.env = env
        self.state = state
        self.memory = None
        self.expert_memory = NOne
        self.eps_len = 0
        self.args = args
        self.rewards = []
        self.entropies = []
        self.ces = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        self.state = self.state.unsqueeze(0)
        _, mu, sigma, self.memory = self.model(
            (Variable(self.state), self.memory))

        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = Variable(eps).cuda()
                pi = Variable(pi).cuda()
        else:
            eps = Variable(eps)
            pi = Variable(pi)

        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        action = (mu + sigma.sqrt() * eps).data
        act = Variable(action)
        prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
        action = torch.clamp(action, -1.0, 1.0)

        _, expert_mu, expert_sigma, self.expert_memory = self.expert(
            (Variable(self.state), self.expert_memory))
        expert_mu = torch.clamp(expert_mu, -1.0, 1.0)
        expert_sigma = F.softplus(expert_sigma) + 1e-5

        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy)

        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        ce = 0.5*torch.log(sigma/expert_sigma) + (expert_sigma + (expert_mu - mu)**2)/(2*expert_sigma) - 0.5
        self.ces.append(ce)

        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.memory = self.model.initialize_memory()
                else:
                    self.memory = self.model.initialize_memory()
            else:
                self.memory = self.model.reinitialize_memory(self.memory)
            self.state = self.state.unsqueeze(0)
            _, mu, sigma, self.memory = self.model(
                (Variable(self.state), self.memory))
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.rewards = []
        self.entropies = []
        self.ces = []
        return self
