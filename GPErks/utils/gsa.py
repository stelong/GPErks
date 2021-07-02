from itertools import combinations
from typing import List, Optional

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.special import binom
from scipy.stats import norm

from GPErks.constants import N_DRAWS, THRESHOLD, N
from GPErks.emulator import GPEmulator
from GPErks.plots.plot_gsa import boxplot, donut, heatmap, network
from GPErks.utils.design import get_minmax


class SobolGSA:
    def __init__(
        self,
        emulator: GPEmulator,
        n: int = N,
        seed: Optional[int] = None,
        xlabels: Optional[List[str]] = None,
        ylabel: Optional[str] = None,
    ):
        self.emulator = emulator
        self.n = n
        self.seed = seed

        self.d = self.emulator.scaled_data.input_size
        self.index_i = (
            xlabels
            if xlabels is not None
            else [f"p_${i+1}$" for i in range(self.d)]
        )
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = ylabel if ylabel is not None else "Output"
        self.minmax = get_minmax(
            self.emulator.scaled_data.scx.inverse_transform(
                self.emulator.scaled_data.X_train.cpu().numpy()
            )
        )  # porcata atomica, questo perche' X_train, y_train originali non sono GPExperiment class attributes

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
            skip_values=0,
        )  # X will have n * (2*d + 2) rows
        return problem, X

    def estimate_Sobol_indices(self, n_draws: int = N_DRAWS):
        problem, X = self.assemble_Saltelli_space()
        Y = self.emulator.sample(X, n_draws)

        conf_level = 0.95
        z = norm.ppf(
            0.5 + conf_level / 2
        )  # trick here to make SALib return plain and NOT scaled std

        for y in Y:
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=100,
                conf_level=conf_level,
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
                (self.ST_std, T_Si["ST_conf"].reshape(1, -1) / z)
            )
            self.S1_std = np.vstack(
                (self.S1_std, first_Si["S1_conf"].reshape(1, -1) / z)
            )
            self.S2_std = np.vstack(
                (
                    self.S2_std,
                    (np.array(second_Si["S2_conf"]).reshape(1, -1) / z),
                )
            )

    def correct_Sobol_indices(self, threshold: float = THRESHOLD):
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def plot(self, kind: str = "boxplot"):
        if kind == "boxplot":
            fun = boxplot
        elif kind == "donut":
            fun = donut
        elif kind == "heatmap":
            fun = heatmap
        elif kind == "network":
            fun = network
        else:
            raise ValueError(
                "Not a valid plotting format! Available formats are: 'boxplot', 'donut', 'heatmap, 'network'."
            )

        fun(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )
