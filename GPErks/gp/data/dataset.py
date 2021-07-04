from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from GPErks.constants import HEIGHT, WIDTH


class Dataset:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        xlabels: Optional[List[str]] = None,
        ylabel: Optional[str] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.with_val = X_val is not None and y_val is not None

        self.input_size = self.X_train.shape[1]

        self.xlabels = (
            xlabels
            if xlabels is not None
            else [f"p_${i+1}$" for i in range(self.input_size)]
        )
        self.ylabel = ylabel if ylabel is not None else "Output"

    def plot(self):
        self.plot_train()

    def plot_pairwise(self):
        self._plot_pairwise(self.X_train)

    def plot_train(self):
        self._plot(self.X_train, self.y_train)

    def plot_test(self):
        self._plot(self.X_test, self.y_test)

    def plot_val(self):
        self._plot(self.X_val, self.y_val)

    def _plot(self, X, y):
        self.n = X.shape[0]
        self.d = X.shape[1]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=self.d,
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 5),
        )
        for i, axis in enumerate(axes.flat):
            axis.scatter(X[:, i], y, facecolor="C0", edgecolor="C0")
            axis.set_xlabel(self.xlabels[i], fontsize=12)
        axes[0].set_ylabel(self.ylabel, fontsize=12)

        plt.suptitle(f"Sample dimension = {self.n} points", fontsize=12)
        plt.show()

    def _plot_pairwise(self, X):
        self.n = X.shape[0]
        self.d = X.shape[1]

        fig, axes = plt.subplots(
            nrows=self.d,
            ncols=self.d,
            sharex="col",
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 2),
        )
        for t, axis in enumerate(axes.flat):
            i = t % self.d
            j = t // self.d
            if j >= i:
                axis.scatter(X[:, i], X[:, j], facecolor="C0", edgecolor="C0")
            else:
                axis.set_axis_off()
            if i == 0:
                axis.set_ylabel(self.xlabels[j])
            if j == self.d - 1:
                axis.set_xlabel(self.xlabels[i])

        plt.suptitle(f"Sample dimension = {self.n} points", fontsize=12)
        plt.show()
