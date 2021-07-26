from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.special import binom

from GPErks.constants import (
    CONF_LEVEL,
    N_BOOTSTRAP,
    N_DRAWS,
    SKIP_VALUES,
    THRESHOLD,
    N,
    Z,
)
from GPErks.plot.gsa import boxplot, donut, heatmap, network
from GPErks.plot.options import PlotOptions
from GPErks.plot.plottable import Plottable
from GPErks.train.emulator import GPEmulator
from GPErks.utils.array import get_minmax


class SobolGSA(Plottable):
    def __init__(
        self,
        emulator: GPEmulator,
        n: int = N,
        seed: Optional[int] = None,
        plot_options: PlotOptions = PlotOptions(),
    ):
        super(SobolGSA, self).__init__(plot_options)
        self.emulator = emulator
        self.n = n
        self.seed = seed

        self.d = self.emulator.experiment.dataset.input_size
        self.index_i = self.emulator.experiment.dataset.x_labels
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = self.emulator.experiment.dataset.y_label
        self.minmax = get_minmax(self.emulator.experiment.dataset.X_train)

        self.ST = np.zeros((0, self.d), dtype=float)
        self.S1 = np.zeros((0, self.d), dtype=float)
        self.S2 = np.zeros((0, int(binom(self.d, 2))), dtype=float)

        self.ST_std = np.zeros((0, self.d), dtype=float)
        self.S1_std = np.zeros((0, self.d), dtype=float)
        self.S2_std = np.zeros((0, int(binom(self.d, 2))), dtype=float)

    def assemble_Saltelli_space(self):
        problem = {
            "num_vars": self.d,
            "names": self.index_i,
            "bounds": self.minmax,
        }
        X = saltelli.sample(
            problem,
            self.n,
            calc_second_order=True,
            skip_values=SKIP_VALUES,
        )
        return problem, X

    def estimate_Sobol_indices(self, n_draws: int = N_DRAWS):
        problem, X = self.assemble_Saltelli_space()
        Y = self.emulator.sample(X, n_draws)

        for y in Y:
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=N_BOOTSTRAP,
                conf_level=CONF_LEVEL,
                parallel=False,
                n_processors=None,
                seed=self.seed,
            )
            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            self.ST = np.vstack((self.ST, T_Si["ST"].reshape(1, -1)))
            self.S1 = np.vstack((self.S1, first_Si["S1"].reshape(1, -1)))
            self.S2 = np.vstack(
                (self.S2, np.array(second_Si["S2"]).reshape(1, -1))
            )

            self.ST_std = np.vstack(
                (self.ST_std, T_Si["ST_conf"].reshape(1, -1) / Z)
            )
            self.S1_std = np.vstack(
                (self.S1_std, first_Si["S1_conf"].reshape(1, -1) / Z)
            )
            self.S2_std = np.vstack(
                (
                    self.S2_std,
                    (np.array(second_Si["S2_conf"]).reshape(1, -1) / Z),
                )
            )

    def correct_Sobol_indices(self, threshold: float = THRESHOLD):
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def summary(self):
        self.correct_Sobol_indices()
        df_STi = pd.DataFrame(
            data=np.round(np.median(self.ST, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["STi"],
        )
        df_Si = pd.DataFrame(
            data=np.round(np.median(self.S1, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["Si"],
        )
        df_Sij = pd.DataFrame(
            data=np.round(np.median(self.S2, axis=0), 6).reshape(-1, 1),
            index=[
                "(" + elem[0] + ", " + elem[1] + ")" for elem in self.index_ij
            ],
            columns=["Sij"],
        )
        print(df_STi)
        print(df_Si)
        print(df_Sij)

    def plot(self):
        self.plot_boxplot()

    def plot_boxplot(self):
        boxplot(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )

    def plot_donut(self):
        donut(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_heatmap(self):
        heatmap(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_network(self):
        network(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )
