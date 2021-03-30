import torch


def tensorize(X):
    return torch.from_numpy(X).float()
