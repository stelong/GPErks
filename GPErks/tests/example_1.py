#!/usr/bin/env python3
import os
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from scipy.stats import qmc
from torchmetrics import MeanSquaredError, R2Score

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.perks.inference import Inference
from GPErks.serialization.path import posix_path
from GPErks.train.early_stop import (
    NoEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
)
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import forrester

log = get_logger()


def main():
    seed = 8
    set_seed(seed)  # reproducible sampling

    # function to learn
    f = lambda X: np.array([forrester(x) for x in X])

    # build dataset
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    train_sample_size = 15
    X_train = sampler.random(n=train_sample_size).ravel()
    y_train = f(X_train)

    test_sample_size = 100
    X_test = np.linspace(0, 1, test_sample_size)
    y_test = f(X_test)

    dataset = Dataset(X_train, y_train, X_test=X_test, y_test=y_test)

    # define experiment options
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = 1
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=input_size))

    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=2,
        metrics=metrics,
        seed=seed,  # reproducible training
    )

    # define training options
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_epochs = 100
    # early_stopping_criterion = NoEarlyStoppingCriterion(max_epochs)
    early_stopping_criterion = PkEarlyStoppingCriterion(
        max_epochs, alpha=0.01, patience=8, strip_length=20
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

    # plot graphs
    best_train_stats.plot()

    # test model
    inference = Inference(emul)
    inference.summary()
    inference.plot()


if __name__ == "__main__":
    main()
