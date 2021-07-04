#!/usr/bin/env python3
import os
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from sklearn.model_selection import train_test_split
from torchmetrics import MeanSquaredError, R2Score

from GPErks.gp.experiment import GPExperiment
from GPErks.plot.training import plot_training_stats
from GPErks.train.early_stop import GLEarlyStoppingCriterion
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.design import read_labels
from GPErks.utils.diagnostics import Diagnostics
from GPErks.utils.gsa import SobolGSA
from GPErks.utils.inference import Inference
from GPErks.utils.log import get_logger
from GPErks.utils.path import posix_path
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import forrester

log = get_logger()


def main():
    seed = 8
    set_seed(seed)  # reproducible sampling

    # function to learn
    f = lambda X: np.array([forrester(x) for x in X])

    # load dataset
    X_train = np.loadtxt("data/nkmodel/X_train.txt", dtype=float)
    y_train = np.loadtxt("data/nkmodel/y_train.txt", dtype=float)

    X_ = np.loadtxt("data/nkmodel/X_test.txt", dtype=float)
    y_ = np.loadtxt("data/nkmodel/y_test.txt", dtype=float)

    X_test, X_val, y_test, y_val = train_test_split(
        X_, y_, test_size=0.5, random_state=seed
    )

    # define experiment options
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = X_train.shape[1]
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=input_size))

    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        X_train,
        y_train,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        X_val=X_val,
        y_val=y_val,
        seed=seed,  # reproducible training
    )

    # define training options
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_epochs = 1000
    # early_stopping_criterion = NoEarlyStoppingCriterion(max_epochs)
    early_stopping_criterion = GLEarlyStoppingCriterion(
        max_epochs, alpha=1.0, patience=8
    )

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
    plot_training_stats(best_train_stats)

    # test model: diagnostics
    diagnostics = Diagnostics(emul, X_test, y_test)
    diagnostics.summary()
    diagnostics.plot()

    # test model: inference
    inference = Inference(emul, X_test, y_test, metrics)
    inference.summary()
    inference.plot()

    # gsa
    path_to_labels_dir = "data/nkmodel/"
    xlabels = read_labels(path_to_labels_dir + "xlabels.txt")
    ylabels = read_labels(path_to_labels_dir + "ylabels.txt")
    idx_emulated_feature = 0
    ylabel = ylabels[idx_emulated_feature]

    gsa = SobolGSA(
        emul, n=1024, seed=seed, xlabels=xlabels, ylabel=ylabel
    )  # non-reproducible results even if setting seed: WHY?!
    gsa.estimate_Sobol_indices(n_draws=1000)

    gsa.correct_Sobol_indices(threshold=0.01)
    gsa.plot(kind="boxplot")
    gsa.plot(kind="donut")
    gsa.plot(kind="heatmap")
    gsa.plot(kind="network")


if __name__ == "__main__":
    main()
