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
from GPErks.perks.cross_validation import KFoldCrossValidation
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
from GPErks.utils.concurrency import execute_task_in_parallel
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import forrester

log = get_logger()


def main():
    ##========================================================================
    ## enforce reproducibility
    ##========================================================================
    seed = 8
    set_seed(seed)

    # ##========================================================================
    # ## load dataset
    # ##========================================================================
    path_to_data = "../../notebooks/data/"
    # X = np.loadtxt(path_to_data + "X.txt", dtype=float)[:100]
    # y = np.loadtxt(path_to_data + "y.txt", dtype=float)[:100]
    # X_, X_test, y_, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=seed
    # )
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_, y_, test_size=0.2, random_state=seed
    # )
    xlabels = read_labels_from_file(path_to_data + "xlabels.txt")
    ylabel = read_labels_from_file(path_to_data + "ylabel.txt")[0]

    # dataset = Dataset(
    #     X_train,
    #     y_train,
    #     X_val=X_val,
    #     y_val=y_val,
    #     X_test=X_test,
    #     y_test=y_test,
    #     x_labels=xlabels,
    #     y_label=ylabel,
    # )

    # ##========================================================================
    # ## define experiment options
    # ##========================================================================
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # mean_function = LinearMean(input_size=dataset.input_size)
    # kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))

    # metrics = [R2Score(), MeanSquaredError()]

    # experiment = GPExperiment(
    #     dataset,
    #     likelihood,
    #     mean_function,
    #     kernel,
    #     n_restarts=3,
    #     metrics=metrics,
    #     seed=seed,  # reproducible training
    #     learn_noise=True,
    # )

    # ##========================================================================
    # ## define training options
    # ##========================================================================
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # early_stopping_criterion = SimpleEarlyStoppingCriterion(1000, patience=8)
    # # early_stopping_criterion = PkEarlyStoppingCriterion(
    # #     1000, alpha=0.001, patience=8, strip_length=20
    # # )

    # kf = KFoldCrossValidation(experiment, [device], n_splits=3, max_workers=1)

    # best_model_dct, best_train_stats_dct, test_scores = kf.train(
    #     optimizer,
    #     early_stopping_criterion=early_stopping_criterion,
    #     leftout_is_val=False,
    # )

    # print(test_scores)

    # for i, bts in best_train_stats_dct.items():
    #     bts.plot(with_early_stopping_criterion=True)

    # ----------------
    # AGAIN

    X = np.loadtxt(path_to_data + "X.txt", dtype=float)[-25:]
    y = np.loadtxt(path_to_data + "y.txt", dtype=float)[-25:]

    dataset = Dataset(
        X,
        y,
        x_labels=xlabels,
        y_label=ylabel,
    )

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
        n_restarts=3,
        metrics=metrics,
        seed=seed,  # reproducible training
        learn_noise=True,
    )

    ##========================================================================
    ## define training options
    ##========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

    kf = KFoldCrossValidation(experiment, [device], n_splits=5, max_workers=1)

    best_model_dct, best_train_stats_dct, test_scores = kf.train(
        optimizer,
        esc,
        leftout_is_val=True,
    )

    print(test_scores)

    best_epochs = []
    for i, bts in best_train_stats_dct.items():
        bts.plot(with_early_stopping_criterion=True)
        best_epochs.append(bts.best_epoch)
    print(best_epochs)


if __name__ == "__main__":
    main()
