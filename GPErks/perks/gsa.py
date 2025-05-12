from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import _sensitivity_analysis, sobol_indices, uniform

from GPErks.constants import (
    DEFAULT_GSA_CONF_LEVEL,
    DEFAULT_GSA_N,
    DEFAULT_GSA_N_BOOTSTRAP,
    DEFAULT_GSA_N_DRAWS,
    DEFAULT_GSA_THRESHOLD,
)
from GPErks.gp.data.dataset import Dataset
from GPErks.plot.gsa import barplot, boxplot, donut, heatmap
from GPErks.train.emulator import GPEmulator
from GPErks.utils.array import get_minmax


class SobolGSA:
    """
    Sobol' global sensitivity analysis using Saltelli method and
    integrating emulator uncertainty.
    """

    def __init__(
        self,
        dataset: Dataset,
        n: int = DEFAULT_GSA_N,
        seed: Optional[int] = None,
    ):
        self.n = n
        self.seed = seed
        self.d = dataset.input_size
        self.xlabels = dataset.x_labels
        self.ylabel = dataset.y_label
        self.minmax = (
            get_minmax(dataset.X_train)
            if (dataset.l_bounds is None and dataset.u_bounds is None)
            else np.hstack(
                (
                    np.array(dataset.l_bounds).reshape(-1, 1),
                    np.array(dataset.u_bounds).reshape(-1, 1),
                )
            )
        )

    def estimate_Sobol_indices_with_simulator(
        self,
        f: Callable[[np.ndarray], np.ndarray],
    ):
        indices = sobol_indices(
            func=f,
            n=self.n,
            dists=[uniform(loc=x[0], scale=x[1] - x[0]) for x in self.minmax],
            rng=self.seed,
        )
        self.boot = indices.bootstrap(
            confidence_level=DEFAULT_GSA_CONF_LEVEL,
            n_resamples=DEFAULT_GSA_N_BOOTSTRAP,
        )
        self.ST = indices.total_order.reshape(1, -1)
        self.S1 = indices.first_order.reshape(1, -1)

    def estimate_Sobol_indices_with_emulator(
        self,
        emulator: GPEmulator,
        n_draws: int = DEFAULT_GSA_N_DRAWS,
    ):
        A, B = _sensitivity_analysis.sample_A_B(
            n=self.n,
            dists=[uniform(loc=x[0], scale=x[1] - x[0]) for x in self.minmax],
            rng=self.seed,
        )
        AB = _sensitivity_analysis.sample_AB(A=A, B=B)
        d, d, n = AB.shape
        AB = np.moveaxis(AB, 0, -1).reshape(d, n * d)
        f_all = emulator.sample(np.hstack((A, B, AB)).T, n_draws)
        f_A, f_B, f_AB = f_all[:, :n], f_all[:, n : 2 * n], f_all[:, 2 * n :]
        f_AB = np.moveaxis(f_AB.reshape((-1, n, d)), -1, 0)
        indices = sobol_indices(
            func={"f_A": f_A, "f_B": f_B, "f_AB": f_AB},
            n=self.n,
            rng=self.seed,
        )
        self.ST = indices.total_order
        self.S1 = indices.first_order

    def correct_Sobol_indices(self, threshold: float = DEFAULT_GSA_THRESHOLD):
        for S in [self.ST, self.S1]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def assemble_dataframe(self):
        value = np.concatenate((self.ST.T.ravel(), self.S1.T.ravel()))
        param = np.repeat(self.xlabels, self.S1.shape[0])
        index = np.repeat(["ST", "S1"], len(param))
        param = np.concatenate((param, param))
        self.df = pd.DataFrame.from_dict(
            dict(Parameter=param, Index=index, Value=value)
        )

    def summary(self):
        self.assemble_dataframe()
        df_pivot = pd.pivot_table(
            self.df,
            values="Value",
            index="Parameter",
            columns="Index",
            aggfunc="median",
        )
        df_STi = pd.DataFrame(
            data=np.round(df_pivot["ST"], 6).values.reshape(-1, 1),
            index=df_pivot.index,
            columns=["STi"],
        )
        df_Si = pd.DataFrame(
            data=np.round(df_pivot["S1"], 6).values.reshape(-1, 1),
            index=df_pivot.index,
            columns=["Si"],
        )
        print(df_STi)
        print(df_Si)

    def plot(
        self,
        axis: Optional[matplotlib.axes._axes.Axes] = None,
        type: str = "box",
        colors: str = "tab10",
    ):
        if axis is None:
            raise ValueError("A matplotlib axis must be specified!")

        match type:
            case "bar":
                axis = barplot(axis, self.df, colors)
            case "box":
                axis = boxplot(axis, self.df, colors)
            case "donut":
                axis = donut(axis, self.df, colors)
            case "heat":
                axis = heatmap(axis, self.df, colors)
            case _:
                raise ValueError(
                    "Plot 'type' must be either 'bar', 'box', 'donut', or 'heat'!"
                )
