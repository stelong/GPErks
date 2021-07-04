import numpy as np
import torch


def get_minmax(X):
    minmax = []
    for x in X.T:
        minmax.append([x.min(), x.max()])
    return np.array(minmax)


def tensorize(X):
    return torch.from_numpy(X).float()
