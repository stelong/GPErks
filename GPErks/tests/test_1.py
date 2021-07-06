#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from sklearn.model_selection import train_test_split
from torchmetrics import MeanSquaredError, R2Score

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment, load_experiment_from_config_file
from GPErks.log.logger import get_logger
from GPErks.perks.inference import Inference
from GPErks.serialization.labels import read_labels_from_file
from GPErks.train.early_stop import (
    GLEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
    PQEarlyStoppingCriterion,
    UPEarlyStoppingCriterion,
)
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.random import set_seed

log = get_logger()


def main():
    seed = 88
    set_seed(seed)

    # ================================================================
    # (1) Loading and visualising dataset
    # ================================================================
    # (2) Building example train, test, val sets
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    Y = np.loadtxt(loadpath + "Y.txt", dtype=float)

    target_label_idx = int(sys.argv[2])
    y = np.copy(Y[:, target_label_idx])

    X_, X_test, y_, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_, y_, test_size=0.2, random_state=seed
    )

    xlabels = read_labels_from_file(loadpath + "xlabels.txt")
    ylabel = read_labels_from_file(loadpath + "ylabels.txt")[target_label_idx]

    dataset = Dataset(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        x_labels=xlabels,
        y_label=ylabel,
    )
    # dataset.plot()
    # dataset.plot_pairwise()

    # ================================================================
    # (3) Training GPE
    # ================================================================
    savepath = sys.argv[3].rstrip("/") + "/" + str(target_label_idx) + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = dataset.input_size
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(
        # MaternKernel(ard_num_dims=input_size),
        RBFKernel(ard_num_dims=input_size),
    )
    # metrics = [ExplainedVariance(), MeanSquaredError(), R2Score()]
    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,
    )
    config_file = __file__.replace(".py", ".ini")
    experiment.save_to_config_file(config_file)

    # Alternatively, load from file
    experiment = load_experiment_from_config_file(
        config_file,
        dataset,
    )

    here = os.path.abspath(os.path.dirname(__file__))
    # snapc = NeverSaveSnapshottingCriterion(
    snapc = EveryEpochSnapshottingCriterion(
        here + "/snapshot/test_1/restart_{restart}/",
        "epoch_{epoch}.pth",
    )

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    max_epochs = 1000
    # early_stopping_criterion = NoEarlyStoppingCriterion(max_epochs)  # TODO: investigate if snapshot is required anyway
    # early_stopping_criterion = PkEarlyStoppingCriterion(max_epochs, alpha=1.0, patience=8, strip_length=20)
    early_stopping_criterion = GLEarlyStoppingCriterion(
        max_epochs, alpha=1.0, patience=8
    )
    # early_stopping_criterion = UPEarlyStoppingCriterion(max_epochs, strip_length=5, successive_strips=4)
    # early_stopping_criterion = PQEarlyStoppingCriterion(max_epochs, alpha=1.0, patience=8, strip_length=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emul = GPEmulator(experiment, device)
    best_model, best_train_stats = emul.train(
        optimizer, early_stopping_criterion, snapc
    )
    best_train_stats.plot(overlay_criterion=True)

    # ================================================================
    # (6) Testing trained GPE at new input points (inference)
    # ================================================================
    emul.experiment.dataset.X_test = X_test
    emul.experiment.dataset.y_test = y_test  # user DOES NOT normally provide a testing dataset during training but only later on

    inference = Inference(emul)
    inference.summary()
    # inference.plot()

    r2s = inference.scores_dct["R2Score"]

    if experiment.scaled_data.with_val and not np.isclose(
        r2s, 0.58630764, rtol=1.0e-5
    ):
        log.error("INCORRECT R2Score (with val)")
    if not experiment.scaled_data.with_val and not np.isclose(
        r2s, 0.89883888, rtol=1.0e-5
    ):
        log.error("INCORRECT R2Score")


if __name__ == "__main__":
    main()
