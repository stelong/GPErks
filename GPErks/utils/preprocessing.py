import numpy as np


class StandardScaler:
    def __init__(self, log_transform=False):
        self.log_transform = log_transform
        self.mu = None
        self.sigma = None

    def fit(self, y):
        if self.log_transform:
            y = np.log(y)
        self.mu = np.mean(y)
        self.sigma = np.std(y)

    def transform(self, y):
        if self.log_transform:
            y = np.log(y)
        y_ = (y - self.mu) / self.sigma
        return y_

    def inverse_transform(self, y_, ystd_=0):
        y = self.sigma * y_ + self.mu
        ystd = self.sigma * ystd_
        if self.log_transform:
            return (
                np.exp(y + 0.5 * np.power(ystd, 2)),
                (np.exp(np.power(ystd, 2)) - 1)
                * np.exp(2 * y + np.power(ystd, 2)),
            )
        else:
            return y, ystd


class UnitCubeScaler:
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, X):
        self.a = np.min(X, axis=0)
        self.b = np.max(X, axis=0)

    def transform(self, X):
        X_ = (X - self.a) / (self.b - self.a)
        return X_

    def inverse_transform(self, X_):
        X = (self.b - self.a) * X_ + self.a
        return X
