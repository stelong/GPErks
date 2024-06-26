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


def delta(X):  # utility function for "part_and_select"
    delta_u_l = [(X[:, i].max() - X[:, i].min()) for i in range(X.shape[1])]
    pj = np.argmax(delta_u_l)
    return pj, delta_u_l[pj]


def part_and_select(P, N):
    C1 = P
    p1, s1 = delta(C1)
    archive = [(s1, p1, C1)]
    i = 1
    while i < N:
        if all([not x[0] for x in archive]):
            break
        sj, pj, Cj = max(archive[:i], key=lambda x: x[0])
        upj = np.max(Cj[:, pj])
        lpj = np.min(Cj[:, pj])
        Cj1 = np.array([x for x in Cj if x[pj] <= (upj + lpj) / 2])
        Cj2 = np.array([x for x in Cj if x[pj] > (upj + lpj) / 2])
        pj1, sj1 = delta(Cj1)
        pj2, sj2 = delta(Cj2)
        archive.remove((sj, pj, Cj))
        archive.append((sj1, pj1, Cj1))
        archive.append((sj2, pj2, Cj2))
        i += 1
    parts = [C for _, _, C in archive]
    selected = []
    for C in parts:
        c = np.mean(C, axis=0)
        idx = np.argmin(np.linalg.norm(C - c, axis=1))
        selected.append(C[idx])
    return parts, np.stack(selected)


def matrix_subtraction(X, XS):
    set_X = set(map(tuple, X))
    set_XS = set(map(tuple, XS))
    return np.array(list(set_X - set_XS))


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
