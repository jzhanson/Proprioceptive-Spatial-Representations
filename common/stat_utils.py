from __future__ import print_function

import scipy.stats
import numpy as np
import numpy.random as npr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def smooth(y, x, std=1, bw=25):
    y_smoothed = np.zeros(y.shape)
    l = len(x)
    for i in range(0,l):
        if i-bw < 0:
            start = 0
        else:
            start = i - bw
        if i+bw > l:
            end = l
        else:
            end = i+bw
        x_win = x[start:end]
        y_win = y[start:end]
        weights = scipy.stats.norm(x[i],std).pdf(x_win)
        y_smoothed[i] = sum(np.multiply(weights, y_win))/sum(weights)
    return y_smoothed
