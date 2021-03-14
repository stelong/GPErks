import numpy as np


def get_minmax(X):
    n_features = X.shape[1]
    return np.hstack(
        (
            np.array([np.min(X[:, j]) for j in range(n_features)]).reshape(
                -1, 1
            ),
            np.array([np.max(X[:, j]) for j in range(n_features)]).reshape(
                -1, 1
            ),
        )
    )


def lhd(I, n_samples):
    n_params = I.shape[0]
    H = np.zeros((n_samples, n_params))
    for j in range(n_params):
        u = np.random.random_sample(n_samples)
        b = np.arange(n_samples)
        np.random.shuffle(b)
        d = (b + u) / n_samples
        H[:, j] = I[j, 0] + (I[j, 1] - I[j, 0]) * d
    return H


def read_labels(name_in):
    labels = []
    with open(name_in, "r") as f:
        for line in f:
            labels.append(line.replace("\n", ""))
    return labels
