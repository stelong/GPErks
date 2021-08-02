#!/usr/bin/env python3
import os
from pathlib import Path

import gpytorch
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
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
    NoEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
    PQEarlyStoppingCriterion,
    SimpleEarlyStoppingCriterion,
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


def main(factor):
    ##========================================================================
    ## reproducible sampling
    ##========================================================================
    seed = 8
    set_seed(seed)

    ##========================================================================
    ## test functions for analytic Sobol' indices
    ##========================================================================
    # D = 3
    # f = lambda X: np.array([Ishigami(x) for x in X])
    # l_bounds, u_bounds = D * [-np.pi], D * [np.pi]
    # df_STi_theo, df_Si_theo, df_Sij_theo = Ishigami_theoretical_Si()

    D = 8
    a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
    delta = np.random.rand(D)
    alpha = np.ones_like(a)
    f = lambda X: np.array([SobolGstar(x, a, delta, alpha) for x in X])
    l_bounds, u_bounds = D * [0], D * [1]
    df_STi_theo, df_Si_theo, df_Sij_theo = SobolGstar_theoretical_Si(
        a, delta, alpha
    )

    ##========================================================================
    ## build training, validation and testing datasets
    ##========================================================================
    n = factor * D
    n_train_samples = n
    n_val_samples = int(n / 4)
    n_test_samples = int(5 * n / 16)

    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples,
        n_val_samples,
        n_test_samples,
        design="lhs",
        seed=seed,
        l_bounds=l_bounds,
        u_bounds=u_bounds,
    )
    dataset.X_val = None
    dataset.y_val = None
    dataset.with_val = False
    # dataset.plot()
    # dataset.plot_pairwise()

    ##========================================================================
    ## define experiment options
    ##========================================================================
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_function = LinearMean(input_size=dataset.input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))

    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=10,
        metrics=metrics,
        seed=seed,
    )

    ##========================================================================
    ## define training options
    ##========================================================================
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_epochs = 1000

    # early_stopping_criterion = SimpleEarlyStoppingCriterion(
    #     max_epochs, patience=20
    # )
    # early_stopping_criterion = GLEarlyStoppingCriterion(
    #     max_epochs, alpha=0.0, patience=20
    # )
    # early_stopping_criterion = UPEarlyStoppingCriterion(
    #     max_epochs, strip_length=5, successive_strips=4
    # )
    # early_stopping_criterion = PQEarlyStoppingCriterion(
    #     max_epochs, alpha=0.1, patience=8, strip_length=5
    # )
    early_stopping_criterion = PkEarlyStoppingCriterion(
        max_epochs, alpha=0.001, patience=8, strip_length=20
    )
    # early_stopping_criterion = NoEarlyStoppingCriterion(max_epochs)

    here = os.path.abspath(os.path.dirname(__file__))
    example_name = Path(__file__).name.replace(".py", "")
    snapshot_dir = posix_path(here, "snapshot", example_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshotting_criterion = EveryEpochSnapshottingCriterion(
        posix_path(snapshot_dir, "restart_{restart}"), "epoch_{epoch}.pth"
    )

    ##========================================================================
    ## train model
    ##========================================================================
    emul = GPEmulator(experiment, device)
    best_model, best_train_stats = emul.train(
        optimizer,
        early_stopping_criterion,
        snapshotting_criterion,
    )

    ##========================================================================
    ## training stats, diagnostics, inference
    ##========================================================================
    # best_train_stats.plot(with_early_stopping_criterion=True)

    # diagnostics = Diagnostics(emul)
    # diagnostics.summary()
    # diagnostics.plot()

    # inference = Inference(emul)
    # inference.summary()
    # inference.plot()

    ##========================================================================
    ## gsa: analytic solution vs GPE-based GSA
    ##========================================================================
    gsa = SobolGSA(dataset, n=2 * 1024, seed=seed)
    gsa.estimate_Sobol_indices_with_emulator(emul, n_draws=1000)
    # gsa.estimate_Sobol_indices_with_simulator(f)
    # gsa.correct_Sobol_indices(threshold=0.01)
    # gsa.plot()
    # gsa.summary()

    # print("\n================\n")
    # print(df_STi_theo)
    # print(df_Si_theo)
    # print(df_Sij_theo)

    df_ST = pd.DataFrame(data=gsa.ST, columns=gsa.index_i)
    df_S1 = pd.DataFrame(data=gsa.S1, columns=gsa.index_i)
    df_S2 = pd.DataFrame(
        data=gsa.S2,
        columns=[
            "(" + elem[0] + ", " + elem[1] + ")" for elem in gsa.index_ij
        ],
    )

    plt.style.use("seaborn")
    gs = grsp.GridSpec(2, 2)
    fig = plt.figure(figsize=(2 * WIDTH, 2 * HEIGHT / 2))

    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(ax=ax0, data=df_S1)
    ax0.set_ylim(0, 1)
    ax0.set_title("First-order effect", fontweight="bold", fontsize=12)
    ax0.set_xticklabels(
        ax0.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l0 = list(df_Si_theo["Si"])
    trans0 = ax0.get_xaxis_transform()
    for k, val in enumerate(l0):
        ax0.axhline(val, c="r", lw=1, ls="--")

    ax1 = fig.add_subplot(gs[0, 1])
    sns.boxplot(ax=ax1, data=df_ST)
    ax1.set_ylim(0, 1)
    ax1.set_title("Total effect", fontweight="bold", fontsize=12)
    ax1.set_xticklabels(
        ax1.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l1 = list(df_STi_theo["STi"])
    trans1 = ax1.get_xaxis_transform()
    for k, val in enumerate(l1):
        ax1.axhline(val, c="r", lw=1, ls="--")

    ax2 = fig.add_subplot(gs[1, :])
    sns.boxplot(ax=ax2, data=df_S2)
    ax2.set_ylim(0, 1)
    ax2.set_title("Second-order effect", fontweight="bold", fontsize=12)
    ax2.set_xticklabels(
        ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l2 = list(df_Sij_theo["Sij"])
    trans2 = ax2.get_xaxis_transform()
    for k, val in enumerate(l2):
        ax2.axhline(val, c="r", lw=1, ls="--")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    factor = 20
    main(factor)
