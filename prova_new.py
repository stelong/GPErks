import timeit
from itertools import combinations
from typing import Callable

import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import norm
from GPErks.utils.test_functions_gsa import (
    Ishigami,
    Ishigami_theoretical_Si,
    SobolGstar,
    SobolGstar_theoretical_Si,
)


np.set_printoptions(formatter={"all": lambda x: "{:.4f}".format(x)})


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = timeit.default_timer()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[{}]".format(self.name))
        print("Elapsed: {:g}".format(timeit.default_timer() - self.tstart))


def fun(N):
    SEED = 8
    np.random.seed(SEED)
    # N = 1000

    # D = 3
    # I = np.array([[-np.pi, np.pi] for _ in range(D)])
    # X = saltelli_space(N, D, SEED, scaling=I)
    # y = evaluate(N, D, lambda X: Ishigami(X), X)

    D = 10
    X = saltelli_space(N, D, SEED)
    a = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4])
    delta = np.random.rand(D)
    alpha = np.ones_like(a)

    with Timer("GPE evaluate"):
        y = evaluate(N, D, lambda X: SobolGstar(X, a, delta, alpha), X)

    with Timer("GSA performance"):
        (
            si,
            si_bootstrap,
            sti,
            sti_bootstrap,
            sij,
            sij_bootstrap,
        ) = compute_Sobol_indices(N, D, y, 1000)

    return si, si_bootstrap, sti, sti_bootstrap, sij, sij_bootstrap


# print(si)


# [GPE evaluate]
# Elapsed: 0.00203988
# [GSA performance]
# Elapsed: 1.17344
# [0.1083 0.1148 0.0995 0.0789 0.0471 0.0294 0.0298 0.0051 0.0054 0.0038]


def main():
    N = np.logspace(2, 3, 7)

    width = 5.91667
    height = 9.36111

    # D = 3
    # df_STi_theo, df_Si_theo, df_Sij_theo = Ishigami_theoretical_Si()

    D = 10
    a = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4])
    delta = np.random.rand(D)
    alpha = np.ones_like(a)
    df_STi_theo, df_Si_theo, df_Sij_theo = SobolGstar_theoretical_Si(
        a, delta, alpha
    )

    index_i = [f"X{i + 1}" for i in range(D)]
    index_ij = [f"({c[0]}, {c[1]})" for c in combinations(index_i, 2)]

    S1 = np.zeros((0, D), dtype=float)
    S1_bootstrap = np.zeros((0, D), dtype=float)
    ST = np.zeros((0, D), dtype=float)
    ST_bootstrap = np.zeros((0, D), dtype=float)
    S2 = np.zeros((0, len(index_ij)), dtype=float)
    S2_bootstrap = np.zeros((0, len(index_ij)), dtype=float)

    for n in N:
        si, si_bootstrap, sti, sti_bootstrap, sij, sij_bootstrap = fun(int(n))
        S1 = np.vstack((S1, si.reshape(1, -1)))
        S1_bootstrap = np.vstack((S1_bootstrap, si_bootstrap.reshape(1, -1)))
        ST = np.vstack((ST, sti.reshape(1, -1)))
        ST_bootstrap = np.vstack((ST_bootstrap, sti_bootstrap.reshape(1, -1)))
        S2 = np.vstack((S2, sij.reshape(1, -1)))
        S2_bootstrap = np.vstack((S2_bootstrap, sij_bootstrap.reshape(1, -1)))

    # ----------------------------------------------------------------

    plt.style.use("seaborn")

    x = [np.log10(n) for n in N]

    fig, axes = plt.subplots(2, 5, figsize=(5 * 2 * width / 5, 2 * height / 4))

    for i, (axis, val) in enumerate(zip(axes.flat, list(df_Si_theo["Si"]))):
        axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
        axis.errorbar(
            x, S1[:, i], yerr=1.96 * S1_bootstrap[:, i], fmt="o", color="C0"
        )
        axis.plot(x, S1[:, i], lw=0.8, c="C0")
        axis.set_xticklabels(
            [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
        )
        axis.set_xticks(np.log10([100, 1000, 10000]))
        lb, ub = axis.get_ylim()
        axis.set_ylim([-0.2, 1])
        axis.set_xlabel("N", fontsize=12)
        axis.set_ylabel(f"X$_{i + 1}$", fontsize=12)

    plt.suptitle("First-order effect", fontsize=12, fontweight="bold")
    fig.tight_layout()
    # plt.savefig("SobolGStar_S1.pdf", bbox_inches="tight", dpi=1000)
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(5 * 2 * width / 5, 2 * height / 4))

    for i, (axis, val) in enumerate(zip(axes.flat, list(df_STi_theo["STi"]))):
        axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
        axis.errorbar(
            x, ST[:, i], yerr=1.96 * ST_bootstrap[:, i], fmt="o", color="C0"
        )
        axis.plot(x, ST[:, i], lw=0.8, c="C0")
        axis.set_xticklabels(
            [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
        )
        axis.set_xticks(np.log10([100, 1000, 10000]))
        lb, ub = axis.get_ylim()
        axis.set_ylim([-0.2, 1])
        axis.set_xlabel("N", fontsize=12)
        axis.set_ylabel(f"X$_{i + 1}$", fontsize=12)

    plt.suptitle("Total effect", fontsize=12, fontweight="bold")
    fig.tight_layout()
    # plt.savefig("SobolGStar_ST.pdf", bbox_inches="tight", dpi=1000)
    plt.show()

    fig, axes = plt.subplots(
        9, 5, figsize=(5 * 2 * width / 5, 4.5 * 2 * height / 4)
    )

    for i, (axis, val) in enumerate(zip(axes.flat, list(df_Sij_theo["Sij"]))):
        axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
        axis.errorbar(
            x, S2[:, i], yerr=1.96 * S2_bootstrap[:, i], fmt="o", color="C0"
        )
        axis.plot(x, S2[:, i], lw=0.8, c="C0")
        axis.set_xticklabels(
            [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
        )
        axis.set_xticks(np.log10([100, 1000, 10000]))
        lb, ub = axis.get_ylim()
        axis.set_ylim([-0.2, 1])
        axis.set_xlabel("N", fontsize=12)
        axis.set_ylabel(index_ij[i], fontsize=12)

    plt.suptitle("Second-order effect", fontsize=12, fontweight="bold")
    fig.tight_layout()
    # plt.savefig("SobolGStar_S12.pdf", bbox_inches="tight", dpi=1000)
    plt.show()


if __name__ == "__main__":
    main()
