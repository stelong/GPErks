from typing import Callable, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc

from GPErks.constants import HEIGHT, WIDTH
from GPErks.plot.options import PlotOptions
from GPErks.plot.plottable import Plottable


class Dataset(Plottable):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        x_labels: Optional[List[str]] = None,
        y_label: Optional[str] = None,
        plot_options: PlotOptions = PlotOptions(),
    ):
        super(Dataset, self).__init__(plot_options)
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_val: Optional[np.ndarray] = X_val
        self.y_val: Optional[np.ndarray] = y_val
        self.X_test: Optional[np.ndarray] = X_test
        self.y_test: Optional[np.ndarray] = y_test
        self.with_val: bool = X_val is not None and y_val is not None

        self.sample_size = self.X_train.shape[0]
        self.input_size = (
            self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1
        )

        self.x_labels: List[str] = (
            x_labels
            if x_labels
            else [f"p{i+1}" for i in range(self.input_size)]
        )
        self.y_label: str = y_label if y_label else "Output"

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
            axis.set_xlabel(self.x_labels[i], fontsize=12)
        axes[0].set_ylabel(self.y_label, fontsize=12)

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
                axis.set_ylabel(self.x_labels[j])
            if j == self.d - 1:
                axis.set_xlabel(self.x_labels[i])

        plt.suptitle(f"Sample dimension = {self.n} points", fontsize=12)
        plt.show()

    @classmethod
    def build_from_function(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        d: int,
        n_train_samples: int,
        n_val_samples: int,
        n_test_samples: int,
        seed: Optional[int] = None,
        l_bounds: Optional[List[float]] = None,
        u_bounds: Optional[List[float]] = None,
    ):
        sampler = qmc.Sobol(d=d, scramble=False, seed=seed)

        X_train = sampler.random(n_train_samples)
        X_val = sampler.random(n_val_samples)
        X_test = sampler.random(n_test_samples)

        if l_bounds is not None and u_bounds is not None:
            X_train = qmc.scale(X_train, l_bounds, u_bounds)
            X_val = qmc.scale(X_val, l_bounds, u_bounds)
            X_test = qmc.scale(X_test, l_bounds, u_bounds)

        y_train = []
        for x in X_train:
            y_train.append(f(x))
        y_train = np.array(y_train)
        y_val = []
        for x in X_val:
            y_val.append(f(x))
        y_val = np.array(y_val)
        y_test = []
        for x in X_test:
            y_test.append(f(x))
        y_test = np.array(y_test)

        return cls(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
