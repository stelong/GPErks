import json
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc

from GPErks.constants import HEIGHT, WIDTH
from GPErks.plot.options import PlotOptions
from GPErks.plot.plottable import Plottable
from GPErks.utils.sampling import Sampler


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
        l_bounds: Optional[List[float]] = None,
        u_bounds: Optional[List[float]] = None,
        name: Optional[str] = None,
        descr: Optional[str] = None,
    ):
        super(Dataset, self).__init__()
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_val: Optional[np.ndarray] = X_val
        self.y_val: Optional[np.ndarray] = y_val
        self.with_val: bool = X_val is not None and y_val is not None
        self.X_test: Optional[np.ndarray] = X_test
        self.y_test: Optional[np.ndarray] = y_test
        self.with_test: bool = X_test is not None and y_test is not None

        self.sample_size = self.X_train.shape[0]
        self.input_size = self.X_train.shape[1]

        self.x_labels: List[str] = (
            x_labels if x_labels else [f"X{i+1}" for i in range(self.input_size)]
        )
        self.y_label: str = y_label if y_label else "y"

        self.l_bounds: List[float] = l_bounds if l_bounds else np.min(X_train, axis=0)
        self.u_bounds: List[float] = u_bounds if u_bounds else np.max(X_train, axis=0)

        self.name: str = name if name else "TestExperiment"
        self.descr: str = (
            descr if descr else "An example dataset to test GPErks' power!"
        )

    def plot(self, plot_options: PlotOptions = PlotOptions()):
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
        sample_size = X.shape[0]
        self.input_size = X.shape[1]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=self.input_size,
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 5),
        )
        for i, axis in enumerate(axes.flat):
            axis.scatter(X[:, i], y, facecolor="C0", edgecolor="C0")
            axis.set_xlabel(self.x_labels[i], fontsize=12)
        axes[0].set_ylabel(self.y_label, fontsize=12)

        plt.suptitle(
            f"{self.name} - Sample dimension = {sample_size} points", fontsize=12
        )
        plt.show()

    def _plot_pairwise(self, X):
        sample_size = X.shape[0]

        fig, axes = plt.subplots(
            nrows=self.input_size,
            ncols=self.input_size,
            sharex="col",
            sharey="row",
            figsize=(2 * WIDTH, 2 * HEIGHT / 2),
        )
        for t, axis in enumerate(axes.flat):
            i = t % self.input_size
            j = t // self.input_size
            if j >= i:
                axis.scatter(X[:, i], X[:, j], facecolor="C0", edgecolor="C0")
            else:
                axis.set_axis_off()
            if i == 0:
                axis.set_ylabel(self.x_labels[j])
            if j == self.input_size - 1:
                axis.set_xlabel(self.x_labels[i])

        plt.suptitle(
            f"{self.name} - Sample dimension = {sample_size} points", fontsize=12
        )
        plt.show()

    def summary(self):
        msg = f"\n{self.name} dataset loaded."
        if self.descr:
            msg += f'\nNotes from the author:\n"{self.descr}"'
        val_avail = f"Yes (size = {self.X_val.shape[0]})" if self.with_val else "No"
        test_avail = f"Yes (size = {self.X_test.shape[0]})" if self.with_test else "No"
        msg += (
            "\nDataset properties:"
            + f"\n-Input size: {self.input_size}"
            + f"\n-Input parameters: {self.x_labels}"
            + "\n-Output size: 1"
            + f"\n-Output feature: {[self.y_label]}"
            + f"\n-Sample size: {self.sample_size}"
            + f"\n-Discrepancy: {self.discrepancy:.4f}"
            + f"\n-Validation data available: {val_avail}"
            + f"\n-Testing data available: {test_avail}"
        )
        print(msg)

    @property
    def discrepancy(self):
        """Quantifies the distance between the continuous uniform distribution on a
        hypercube and the discrete uniform distribution on distinct sample points.
        The lower the value, the better the coverage of the parameter space
        """
        X_train_unit_cube = qmc.scale(
            self.X_train, self.l_bounds, self.u_bounds, reverse=True
        )
        return qmc.discrepancy(X_train_unit_cube)

    @classmethod
    def build_from_file(
        cls,
        path_to_data: Optional[Union[Path, str]],
    ):
        if isinstance(path_to_data, str):
            path_to_data = Path(path_to_data)
        name = path_to_data.stem

        with open(path_to_data, "r") as f:
            data = json.load(f)

        descr = None
        if "info" in data.keys():
            descr = data["info"]  # e.g., notes from the author about the dataset

        if "X_train" not in data.keys() or "Y_train" not in data.keys():
            raise ValueError("Dataset does not contain a training dataset!")
        X_train = np.array(data["X_train"])
        Y_train = np.array(data["Y_train"]).T

        x_labels, y_labels = None, Y_train.shape[1] * [None]
        if "x_labels" in data.keys():
            x_labels = data["x_labels"]
        if "y_labels" in data.keys():
            y_labels = data["y_labels"]

        l_bounds, u_bounds = None, None
        if "l_bounds" in data.keys():
            x_labels = data["l_bounds"]
        if "u_bounds" in data.keys():
            y_labels = data["u_bounds"]

        X_val, Y_val = None, Y_train.shape[1] * [None]
        if "X_val" in data.keys() and "Y_val" in data.keys():
            X_val = np.array(data["X_val"])
            Y_val = np.array(data["Y_val"]).T

        X_test, Y_test = None, Y_train.shape[1] * [None]
        if "X_test" in data.keys() and "Y_test" in data.keys():
            X_test = np.array(data["X_test"])
            Y_test = np.array(data["Y_test"]).T

        datasets = {
            y_label: cls(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                x_labels=x_labels,
                y_label=y_label,
                l_bounds=l_bounds,
                u_bounds=u_bounds,
                name=f"{name}_{y_label}",
                descr=descr,
            )
            for (y_train, y_val, y_test, y_label) in zip(
                Y_train, Y_val, Y_test, y_labels
            )
        }
        return datasets

    @classmethod
    def build_from_function(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        d: int,
        n_train_samples: int,
        n_val_samples: Optional[int] = None,
        n_test_samples: Optional[int] = None,
        design: str = "lhs",
        seed: Optional[int] = None,
        x_labels: Optional[List[str]] = None,
        y_label: Optional[str] = None,
        l_bounds: Optional[List[float]] = None,
        u_bounds: Optional[List[float]] = None,
    ):
        name = f.__name__
        descr = f"This dataset was generated by using {name} function evaluations."
        sampler = Sampler(design, d, seed)

        X_train = sampler.sample(n_train_samples, l_bounds, u_bounds)
        y_train = np.squeeze(f(X_train))

        if n_val_samples is not None:
            X_val = sampler.sample(n_val_samples, l_bounds, u_bounds)
            y_val = np.squeeze(f(X_val))
        else:
            X_val, y_val = None, None

        if n_test_samples is not None:
            X_test = sampler.sample(n_test_samples, l_bounds, u_bounds)
            y_test = np.squeeze(f(X_test))
        else:
            X_test, y_test = None, None

        return cls(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            x_labels=x_labels,
            y_label=y_label,
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            name=name,
            descr=descr,
        )
