#!/usr/bin/env python3
import os
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from sklearn.model_selection import KFold, train_test_split
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
    UPEarlyStoppingCriterion,
)
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.concurrency import execute_task_in_parallel
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import forrester

log = get_logger()


def main(early_stopping_criterion, X_train, y_train, X_val=None, y_val=None):
    seed = 8
    set_seed(seed)  # reproducible sampling

    ##========================================================================
    ## load dataset
    ##========================================================================
    path_to_data = "data/nkmodel/"
    X_test = np.loadtxt(path_to_data + "X_test.txt", dtype=float)
    y_test = np.loadtxt(path_to_data + "y_test.txt", dtype=float)

    target_label_idx = 0
    xlabels = read_labels_from_file(path_to_data + "xlabels.txt")
    ylabel = read_labels_from_file(path_to_data + "ylabels.txt")[
        target_label_idx
    ]

    if X_val is not None and y_val is not None:
        dataset = Dataset(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            x_labels=xlabels,
            y_label=ylabel,
        )
    else:
        dataset = Dataset(
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            x_labels=xlabels,
            y_label=ylabel,
        )

    ##========================================================================
    ## define experiment options
    ##========================================================================
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
        n_restarts=1,
        metrics=metrics,
        seed=seed,  # reproducible training
    )

    ##========================================================================
    ## define training options
    ##========================================================================
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    here = os.path.abspath(os.path.dirname(__file__))
    example_name = Path(__file__).name.replace(".py", "")
    snapshot_dir = posix_path(here, "snapshot", example_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshotting_criterion = EveryEpochSnapshottingCriterion(
        posix_path(snapshot_dir, "restart_{restart}"),
        "epoch_{epoch}.pth",
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

    inference = Inference(emul)
    inference.summary()

    return best_train_stats.best_epoch


if __name__ == "__main__":
    path_to_data = "data/nkmodel/"
    X = np.loadtxt(path_to_data + "X_train.txt", dtype=float)
    y = np.loadtxt(path_to_data + "y_train.txt", dtype=float)

    FOLD = 5
    kf = KFold(n_splits=FOLD, shuffle=False, random_state=None)

    max_epochs = 500
    early_stopping_criterion = UPEarlyStoppingCriterion(
        max_epochs,
        strip_length=5,
        successive_strips=4,
    )

    inputs = {
        split: (
            early_stopping_criterion,
            X[idx_train],
            y[idx_train],
            X[idx_val],
            y[idx_val],
        )
        for split, (idx_train, idx_val) in enumerate(kf.split(X))
    }
    outputs = execute_task_in_parallel(main, inputs)

    epochs = [outputs[i] for i in range(FOLD)]
    mean_epoch = int(np.mean(epochs))
    print(mean_epoch, epochs)

    early_stopping_criterion = NoEarlyStoppingCriterion(mean_epoch)
    main(early_stopping_criterion, X, y)
