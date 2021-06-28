#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, \
    PeriodicKernel
from gpytorch.means import LinearMean
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from torchmetrics import ExplainedVariance, MeanSquaredError, R2Score

from GPErks.emulator import GPEmulator
from GPErks.experiment import GPExperiment
from GPErks.snapshotting import (
    EveryEpochSnapshottingCriterion,
    NeverSaveSnapshottingCriterion,
)
from GPErks.utils.design import read_labels
from GPErks.utils.diagnostics import Diagnostics
from GPErks.utils.earlystopping import (
    GLEarlyStoppingCriterion,
    NoEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
)
from GPErks.utils.inference import Inference
from GPErks.utils.log import get_logger
from GPErks.utils.metrics import IndependentStandardError as ISE
from GPErks.utils.random import set_seed
from GPErks.utils.tensor import tensorize
from GPErks.utils.test_functions import currin_exp, forrester

log = get_logger()


def main():
    # ================================================================
    # (0) Making the code reproducible
    # ================================================================
    seed = 8
    set_seed(seed)

    # ================================================================
    # (2) Building example training, validation and test datasets
    # ================================================================
    sampler = qmc.Sobol(d=1, scramble=False, seed=seed)

    # X = sampler.random_base2(m=4)
    # X_train = X[:8].ravel()
    # X_val = X[8:12].ravel()
    # X_test = X[12:].ravel()
    X_train = np.random.rand(10)
    X_val = np.random.rand(4)
    X_test = np.random.rand(25)

    # fig, axis = plt.subplots(1, 1)
    # axis.scatter(X_train[:, 0], X_train[:, 1], fc="C0", ec="C0", label="train")
    # axis.scatter(X_val[:, 0], X_val[:, 1], fc="C1", ec="C1", label="val")
    # axis.scatter(X_test[:, 0], X_test[:, 1], fc="C2", ec="C2", label="test")
    # plt.legend()
    # plt.show()

    f = lambda X: np.array([forrester(x) for x in X])
    y_train = f(X_train)
    y_val = f(X_val)
    y_test = f(X_test)

    # ================================================================
    # (3) Training GPE
    # ================================================================
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(
        RBFKernel(ard_num_dims=input_size) * PeriodicKernel(),
    )
    metrics = [ExplainedVariance(), R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        X_train,
        y_train,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        # X_val=X_val,
        # y_val=y_val,
        seed=seed,
    )
    config_file = __file__.replace(".py", ".ini")
    experiment.save_to_config_file(config_file)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    MAX_EPOCHS = 100
    early_stopping_criterion = PkEarlyStoppingCriterion(
        MAX_EPOCHS, alpha=1.0, patience=8, strip_length=20
    )
    # early_stopping_criterion = GLEarlyStoppingCriterion(
    #     MAX_EPOCHS, alpha=1.0, patience=8
    # )

    here = os.path.abspath(os.path.dirname(__file__))
    snapshotting_criterion = EveryEpochSnapshottingCriterion(
        here + "/snapshot/test_4/restart_{restart}/",
        "epoch_{epoch}.pth",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emul = GPEmulator(experiment, device)
    emul.train(optimizer, early_stopping_criterion, snapshotting_criterion)

    # ================================================================
    # (4) Testing trained GPE at new input points (inference)
    # ================================================================
    D = Diagnostics(emul, X_test, y_test)
    D.summary()
    D.plot()

    I = Inference(emul, X_test, y_test, metrics)
    I.summary()
    I.plot()
    # I.interpolate_2Dgrid(f)  # only works for 2D inputs


if __name__ == "__main__":
    main()
