import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, stats

from GPErks.constants import HEIGHT, WIDTH
from GPErks.train.emulator import GPEmulator


class Diagnostics:
    """
    A set of diagnostics to validate and assess the adequacy of a
    Gaussian process emulator as surrogate for a given computer program.
    Ref: L.S. Bastos and A. O’Hagan (2009) doi:10.1198/TECH.2009.08019
    """

    def __init__(
        self,
        emulator: GPEmulator,
    ):
        self.emulator = emulator
        self.X_test = self.emulator.experiment.dataset.X_test
        self.y_test = self.emulator.experiment.dataset.y_test
        (
            self.y_pred_mean,
            self.y_pred_std,
            self.y_pred_covar,
        ) = self.emulator.predict(self.X_test, with_covar=True)

        self.n = self.emulator.experiment.dataset.sample_size
        self.q = self.emulator.experiment.dataset.input_size
        self.m = (
            self.X_test.shape[0]
            if len(self.X_test.shape) > 1
            else len(self.X_test)
        )

    def chi_squared(self):
        """
        Chi-squared distribution with m degrees of freedom.
        """
        CS_mean = self.m
        CS_std = np.sqrt(2 * self.m)

        CS = DChi2(self.y_test, self.y_pred_mean, self.y_pred_std)

        return CS, CS_mean, CS_std

    def mahalanobis_distance(self):
        """
        Scaled F-Snedecor distribution with m and n-q degrees of freedom.
        """
        MD_mean = self.m
        MD_std = np.sqrt(
            2 * self.m * (self.m + self.n - self.q - 2) / (self.n - self.q - 4)
        )

        MD = DMD(self.y_test, self.y_pred_mean, self.y_pred_covar)

        return MD, MD_mean, MD_std

    def credible_interval(self):
        """
        Not a distribution: Monte Carlo simulations are required for
        mean and std estimates.
        """
        Y = self.emulator.sample(self.X_test)
        ci_samples = []
        for y in Y:
            ci_samples.append(DCI(self.y_test, y, self.y_pred_std))
        CI_mean = np.mean(ci_samples)
        CI_std = np.std(ci_samples)

        CI = DCI(self.y_test, self.y_pred_mean, self.y_pred_std)

        return CI, CI_mean, CI_std

    def summary(self):
        df = pd.DataFrame(
            data=np.around(
                np.array(
                    [
                        list(self.chi_squared()),
                        list(self.mahalanobis_distance()),
                        list(self.credible_interval()),
                    ]
                ),
                decimals=4,
            ),
            index=["Chi-Squared", "Mahalanobis Distance", "Credible Interval"],
            columns=["Observed", "Expected", "Std"],
        )
        print(df)

    def plot(self, uncorrelated: bool = True):
        if uncorrelated:
            errors = list(DG(self.y_test, self.y_pred_mean, self.y_pred_covar))
            ylab = "$D^{G}(y^{*})$"
        else:
            errors = list(DI(self.y_test, self.y_pred_mean, self.y_pred_std))
            ylab = "$D^{I}(y^{*})$"

        plt.rcParams.update({"mathtext.default": "regular"})

        fig, axes = plt.subplots(
            1, 2, sharey="row", figsize=(2 * WIDTH, 2 * HEIGHT / 4)
        )

        axes[0].axhline(-2, c="r", ls="--", lw=0.8)
        axes[0].axhline(0, c="k", ls="--", lw=0.8)
        axes[0].axhline(2, c="r", ls="--", lw=0.8)
        axes[0].scatter(self.y_pred_mean, errors)

        axes[0].set_ylabel(ylab, fontsize=12)
        axes[0].set_xlabel(r"$E[\eta (x_{i}^{*})|y]$", fontsize=12)

        errors.sort()
        theoretical_percentiles = stats.mstats.plotting_positions(errors)
        theoretical_quantiles = stats.norm.ppf(theoretical_percentiles)

        axes[1].scatter(theoretical_quantiles, errors, zorder=2)

        q25 = stats.scoreatpercentile(errors, 25)
        q75 = stats.scoreatpercentile(errors, 75)
        tq = stats.norm.ppf([0.25, 0.75])
        a = (q75 - q25) / np.diff(tq)
        b = q25 - a * tq[0]

        xl, xr = axes[1].get_xlim()
        axes[1].plot(
            [xl, xr],
            [a * xl + b, a * xr + b],
            c="r",
            lw=0.8,
            zorder=1,
            label="Line fit through\n1st and 3rd quartiles",
        )
        axes[1].set_xlim([xl, xr])
        axes[1].legend(loc="lower right")
        axes[1].set_xlabel("Theoretical Quantiles", fontsize=12)

        fig.tight_layout()
        plt.show()


def DI(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
    return (y_true - y_pred_mean) / y_pred_std


def DChi2(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray):
    x = DI(y_true, y_pred_mean, y_pred_std)
    return np.sum(np.power(x, 2))


def DMD(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_covar: np.ndarray):
    x = (y_true - y_pred_mean).reshape(-1, 1)
    G = linalg.cho_factor(y_pred_covar)
    return np.squeeze(np.matmul(x.T, linalg.cho_solve(G, x)))


def DG(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_covar: np.ndarray):
    x = (y_true - y_pred_mean).reshape(-1, 1)
    G = linalg.cholesky(y_pred_covar, lower=True)
    return np.squeeze(linalg.solve(G, x))


def DCI(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    marginal_credible_interval: float = 95.0,
):
    z = stats.norm.ppf(0.5 + 0.01 * marginal_credible_interval / 2)
    x = DI(y_true, y_pred_mean, y_pred_std)
    return len(np.where(np.abs(x) < z)[0]) / len(x)
