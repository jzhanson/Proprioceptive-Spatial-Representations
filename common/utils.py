from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for (pname, param), shared_param in zip(model.named_parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is not None:
                shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvLSTM':
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal(x, mu, sigma, gpu_id, gpu=False):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    if gpu:
        with torch.cuda.device(gpu_id):
            pi = Variable(pi).cuda()
    else:
        pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b

def recenter_old_grid(old_grid, old_anchor, new_grid, new_anchor):
    old_anchor_x, old_anchor_y = old_anchor
    new_anchor_x, new_anchor_y = new_anchor
    drift_x = new_anchor_x - old_anchor_x
    drift_y = new_anchor_y - old_anchor_y

    old_grid_recentered = new_grid.clone()*0.

    start_x, end_x = 0+drift_x, old_grid.size(-2)+drift_x
    start_y, end_y = 0+drift_y, old_grid.size(-1)+drift_y

    old_grid_recentered[...,start_x:end_x,start_y:end_y] = old_grid

    '''
    if drift_x != 0 or drift_y != 0:
        print(old_anchor)
        print(new_anchor)
        print(start_x, end_x)
        print(start_y, end_y)

        print(old_grid[0,0,0])
        print(old_grid_recentered[0,0,0])
        exit()
    '''

    return old_grid_recentered
