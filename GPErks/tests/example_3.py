#!/usr/bin/env python3
import os
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from torchmetrics import MeanSquaredError, R2Score

from GPErks.constants import HEIGHT, WIDTH
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.perks.diagnostics import Diagnostics
from GPErks.perks.gsa import SobolGSA
from GPErks.perks.inference import Inference
from GPErks.serialization.labels import read_labels_from_file
from GPErks.serialization.path import posix_path
from GPErks.train.early_stop import (
    GLEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
    UPEarlyStoppingCriterion,
)
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import forrester
from GPErks.utils.test_functions_gsa import (
    Ishigami,
    Ishigami_theoretical_Si,
    SobolGstar,
    SobolGstar_theoretical_Si,
)

log = get_logger()


def main():
    seed = 8
    set_seed(seed)  # reproducible sampling

    # D = 10
    # a = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4])
    # delta = np.random.rand(D)
    # alpha = np.ones_like(a)
    # f = lambda X: SobolGstar(X, a, delta, alpha)

    D = 3
    l_bounds, u_bounds = D * [-np.pi], D * [np.pi]
    f = lambda X: Ishigami(X)

    sampler = qmc.LatinHypercube(d=D, seed=seed)

    n_train_samples = 20 * D
    n_val_samples = 2 * D
    n_test_samples = 5 * D

    X_train = qmc.scale(sampler.random(n=n_train_samples), l_bounds, u_bounds)
    X_test = qmc.scale(sampler.random(n=n_test_samples), l_bounds, u_bounds)
    X_val = qmc.scale(sampler.random(n=n_val_samples), l_bounds, u_bounds)

    y_train = f(X_train)
    y_test = f(X_test)
    y_val = f(X_val)

    dataset = Dataset(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    dataset.plot()
    dataset.plot_pairwise()

    # define experiment options
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = dataset.input_size
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=input_size))

    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=5,
        metrics=metrics,
        seed=seed,  # reproducible training
    )

    # define training options
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_epochs = 500
    # early_stopping_criterion = GLEarlyStoppingCriterion(max_epochs, alpha=1.0, patience=8)
    early_stopping_criterion = UPEarlyStoppingCriterion(
        max_epochs, strip_length=20, successive_strips=4
    )
    # early_stopping_criterion = PkEarlyStoppingCriterion(
    #     max_epochs, alpha=1.0, patience=8, strip_length=20
    # )

    here = os.path.abspath(os.path.dirname(__file__))
    example_name = Path(__file__).name.replace(".py", "")
    snapshot_dir = posix_path(here, "snapshot", example_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshotting_criterion = EveryEpochSnapshottingCriterion(
        # snapshotting_criterion = EveryNEpochsSnapshottingCriterion(
        posix_path(snapshot_dir, "restart_{restart}"),
        "epoch_{epoch}.pth",
        # 5,
    )

    # train model
    emul = GPEmulator(experiment, device)
    best_model, best_train_stats = emul.train(
        optimizer,
        early_stopping_criterion,
        snapshotting_criterion,
    )

    # plot training statistics (train loss, val loss, metrics)
    best_train_stats.plot(overlay_criterion=True)

    # test model: diagnostics
    diagnostics = Diagnostics(emul)
    diagnostics.summary()
    diagnostics.plot()

    # test model: inference
    inference = Inference(emul)
    inference.summary()
    inference.plot()

    # gsa: analytic solution vs GPE-based GSA
    df_STi_theo, df_Si_theo, df_Sij_theo = Ishigami_theoretical_Si()
    print(df_STi_theo)
    print(df_Si_theo)
    print(df_Sij_theo)

    gsa = SobolGSA(emul, n=1024, seed=seed)
    gsa.estimate_Sobol_indices(n_draws=1000)
    gsa.correct_Sobol_indices(threshold=0.01)
    gsa.plot()

    ##-----------------------------------------------------------------------------
    ## first attempt below (went really bad, need to understand why)

    # def fun(n):
    #     gsa = SobolGSA(emul, n=1024, seed=seed)
    #     gsa.estimate_Sobol_indices()
    #     return np.mean(gsa.ST, axis=0), np.mean(gsa.ST_std, axis=0), np.mean(gsa.S1, axis=0), np.mean(gsa.S1_std, axis=0), np.mean(gsa.S2, axis=0), np.mean(gsa.S2_std, axis=0)

    # ST = np.zeros((0, D), dtype=float)
    # ST_std = np.zeros((0, D), dtype=float)
    # S1 = np.zeros((0, D), dtype=float)
    # S1_std = np.zeros((0, D), dtype=float)
    # S2 = np.zeros((0, len(index_ij)), dtype=float)
    # S2_std = np.zeros((0, len(index_ij)), dtype=float)

    # N = np.logspace(2, 3, 7)

    # for n in N:
    #     sti, sti_std, si, si_std, sij, sij_std = fun(int(n))
    #     ST = np.vstack((ST, sti.reshape(1, -1)))
    #     ST_std = np.vstack((ST_std, sti_std.reshape(1, -1)))
    #     S1 = np.vstack((S1, si.reshape(1, -1)))
    #     S1_std = np.vstack((S1_std, si_std.reshape(1, -1)))
    #     S2 = np.vstack((S2, sij.reshape(1, -1)))
    #     S2_std = np.vstack((S2_std, sij_std.reshape(1, -1)))

    # ## plotting GPE-based GSA vs analytic solution
    # plt.style.use("seaborn")

    # def sci_notation(num, decimal_digits=2, precision=2, exponent=None):
    #     if exponent is None:
    #         exponent = int(np.floor(np.log10(np.abs(num))))
    #     coeff = np.around(num / float(10**exponent), decimal_digits)
    #     if precision is None:
    #         precision = decimal_digits

    #     return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)

    # x = [np.log10(n) for n in N]

    # fig, axes = plt.subplots(1, 3, figsize=(5 * 2 * WIDTH / 5, 2 * HEIGHT / 4))

    # for i, (axis, val) in enumerate(zip(axes.flat, list(df_STi_theo["STi"]))):
    #     axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
    #     axis.errorbar(
    #         x, ST[:, i], yerr=1.96 * ST_std[:, i], fmt="o", color="C0"
    #     )
    #     axis.plot(x, ST[:, i], lw=0.8, c="C0")
    #     axis.set_xticklabels(
    #         [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
    #     )
    #     axis.set_xticks(np.log10([100, 1000, 10000]))
    #     lb, ub = axis.get_ylim()
    #     axis.set_ylim([-0.2, 1])
    #     axis.set_xlabel("N", fontsize=12)
    #     axis.set_ylabel(f"X$_{i + 1}$", fontsize=12)

    # plt.suptitle("Total effect", fontsize=12, fontweight="bold")
    # fig.tight_layout()
    # plt.show()

    # fig, axes = plt.subplots(1, 3, figsize=(5 * 2 * WIDTH / 5, 2 * HEIGHT / 4))

    # for i, (axis, val) in enumerate(zip(axes.flat, list(df_Si_theo["Si"]))):
    #     axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
    #     axis.errorbar(
    #         x, S1[:, i], yerr=1.96 * S1_std[:, i], fmt="o", color="C0"
    #     )
    #     axis.plot(x, S1[:, i], lw=0.8, c="C0")
    #     axis.set_xticklabels(
    #         [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
    #     )
    #     axis.set_xticks(np.log10([100, 1000, 10000]))
    #     lb, ub = axis.get_ylim()
    #     axis.set_ylim([-0.2, 1])
    #     axis.set_xlabel("N", fontsize=12)
    #     axis.set_ylabel(f"X$_{i + 1}$", fontsize=12)

    # plt.suptitle("First-order effect", fontsize=12, fontweight="bold")
    # fig.tight_layout()
    # plt.show()

    # fig, axes = plt.subplots(
    #     1, 3, figsize=(5 * 2 * WIDTH / 5, 4.5 * 2 * HEIGHT / 4)
    # )

    # for i, (axis, val) in enumerate(zip(axes.flat, list(df_Sij_theo["Sij"]))):
    #     axis.axhline(val, c="r", ls="--", lw=0.8, zorder=1)
    #     axis.errorbar(
    #         x, S2[:, i], yerr=1.96 * S2_std[:, i], fmt="o", color="C0"
    #     )
    #     axis.plot(x, S2[:, i], lw=0.8, c="C0")
    #     axis.set_xticklabels(
    #         [f"{sci_notation(n, precision=0)}" for n in [100, 1000, 10000]]
    #     )
    #     axis.set_xticks(np.log10([100, 1000, 10000]))
    #     lb, ub = axis.get_ylim()
    #     axis.set_ylim([-0.2, 1])
    #     axis.set_xlabel("N", fontsize=12)
    #     axis.set_ylabel(index_ij[i], fontsize=12)

    # plt.suptitle("Second-order effect", fontsize=12, fontweight="bold")
    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
