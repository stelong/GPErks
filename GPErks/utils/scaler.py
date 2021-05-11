from abc import ABCMeta, abstractmethod

import numpy as np


class InputDataScaler(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def inverse_transform(self, X_):
        pass


class OutputDataScaler(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, y):
        pass

    @abstractmethod
    def transform(self, y):
        pass

    @abstractmethod
    def inverse_transform(self, y_, ystd_=None):
        pass


class UnitCubeScaler(InputDataScaler):
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


class StandardScaler(OutputDataScaler):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, y):
        self.mu = np.mean(y)
        self.sigma = np.std(y)

    def transform(self, y):
        y_ = (y - self.mu) / self.sigma
        return y_

    def inverse_transform(self, y_, ystd_=None):
        y = self.sigma * y_ + self.mu
        ystd = self.sigma * ystd_ if ystd_ is not None else None
        return y, ystd


class StandardLogScaler(OutputDataScaler):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, y):
        y = np.log(y)
        self.mu = np.mean(y)
        self.sigma = np.std(y)

    def transform(self, y):
        y = np.log(y)
        y_ = (y - self.mu) / self.sigma
        return y_

    def inverse_transform(self, y_, ystd_=None):
        if not ystd_:
            raise ValueError(
                "StandardLogScaler cannot back-transform data without ystd_"
            )
        y = self.sigma * y_ + self.mu
        ystd = self.sigma * ystd_
        return (
            np.exp(y + 0.5 * np.power(ystd, 2)),
            (np.exp(np.power(ystd, 2)) - 1)
            * np.exp(2 * y + np.power(ystd, 2)),
        )
