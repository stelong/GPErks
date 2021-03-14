import numpy as np


def diff(l1, l2):
    return list(set(l1) - set(l2))


def inters(l1, l2):
    return list(set(l1) & set(l2))


def inters_many(L):
    return list(set.intersection(*map(lambda x: set(x), L)))


def union_many(L):
    return list(set.union(*map(lambda x: set(x), L)))


def restrict_kth_comp(data, k, ib, ub):
    l = []
    for i in range(data.shape[0]):
        if (
            np.where(data[i, k] > ib)[0].shape[0]
            and np.where(data[i, k] < ub)[0].shape[0]
        ):
            l.append(i)
    return l


def find_start_seq(index, feat_dim):  # utility function for "whereq_whernot"
    i = 0
    while i < len(index):
        if index[i : feat_dim + i] == list(range(feat_dim)):
            return i
        else:
            i += 1
    return


def whereq_whernot(X, SX):
    feat_dim = X.shape[1]
    l = []
    for i in range(SX.shape[0]):
        index = np.where(X == SX[i, :])
        if len(list(index[1])) > feat_dim:
            l.append(index[0][find_start_seq(list(index[1]), feat_dim)])
        else:
            l.append(index[0][0])
    nl = diff(range(X.shape[0]), l)
    nl.sort()
    return l, nl


def filter_zscore(X, thre):
    feat_dim = X.shape[1]
    L = []
    for j in range(feat_dim):
        z = np.abs((X[:, j] - np.mean(X[:, j])) / np.std(X[:, j]))
        L.append(np.where(z > thre)[0])
    nl = union_many(L)
    l = diff(range(X.shape[0]), nl)
    return l, nl
