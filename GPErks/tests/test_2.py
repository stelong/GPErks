#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from sklearn.model_selection import train_test_split
from torchmetrics import ExplainedVariance, MeanSquaredError, R2Score

from GPErks.emulator import GPEmulator
from GPErks.experiment import GPExperiment
from GPErks.snapshotting import (
    EveryEpochSnapshottingCriterion,
    NeverSaveSnapshottingCriterion,
)
from GPErks.utils.design import read_labels
from GPErks.utils.earlystopping import (
    GLEarlyStoppingCriterion,
    NoEarlyStoppingCriterion,
    PkEarlyStoppingCriterion,
)
from GPErks.utils.log import get_logger
from GPErks.utils.metrics import IndependentStandardError as ISE
from GPErks.utils.random import set_seed
from GPErks.utils.tensor import tensorize

log = get_logger()


def main():
    seed = 88
    set_seed(seed)

    # ================================================================
    # (1) Loading and visualising dataset
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    Y = np.loadtxt(loadpath + "Y.txt", dtype=float)

    # xlabels = read_labels(loadpath + "xlabels.txt")
    ylabels = read_labels(loadpath + "ylabels.txt")
    # plot_dataset(X, Y, xlabels, ylabels)

    # ================================================================
    # (2) Building example training and validation datasets
    # ================================================================
    idx_feature = sys.argv[2]
    print(f"\n{ylabels[int(idx_feature)]} feature selected for emulation.")

    y = np.copy(Y[:, int(idx_feature)])

    X_, X_test, y_, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_, y_, test_size=0.2, random_state=seed
    )

    # ================================================================
    # (3) Training GPE
    # ================================================================
    savepath = sys.argv[3].rstrip("/") + "/" + idx_feature + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    np.savetxt(savepath + "X_train.txt", X_train, fmt="%.6f")
    np.savetxt(savepath + "y_train.txt", y_train, fmt="%.6f")
    np.savetxt(savepath + "X_val.txt", X_val, fmt="%.6f")
    np.savetxt(savepath + "y_val.txt", y_val, fmt="%.6f")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = X_train.shape[1]
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(
        # MaternKernel(ard_num_dims=input_size),
        RBFKernel(ard_num_dims=input_size),
    )
    # metrics = [ExplainedVariance(), MeanSquaredError(), R2Score()]
    metrics = [R2Score(), MeanSquaredError()]

    experiment = GPExperiment(
        X_train,
        y_train,
        likelihood,
        mean_function,
        kernel,
        3,
        metrics=metrics,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
    )

    # snapc = NeverSaveSnapshottingCriterion(
    snapc = EveryEpochSnapshottingCriterion(
        "/home/gianvito/personal/GPErks/GPErks/tests/snapshot/my_model/restart_{restart}/",
        "epoch_{epoch}.pth",
    )

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # esc = NoEarlyStoppingCriterion(33)  # TODO: investigate if snapshot is required anyway
    MAX_EPOCHS = 1000
    esc = GLEarlyStoppingCriterion(MAX_EPOCHS, alpha=1.0, patience=8)
    # esc = PkEarlyStoppingCriterion(
    #     MAX_EPOCHS, alpha=1.0, patience=8, strip_length=20
    # )

    emul = GPEmulator(experiment, optimizer)
    emul.train(esc, snapc, save_losses=True)

    # ================================================================
    # (4) Saving trained GPE
    # ================================================================
    # emul.save()

    # ================================================================
    # (5) Loading already trained GPE
    # ================================================================
    # NOTE: you need exactly the same training dataset used in (3)
    # ================================================================
    # loadpath = savepath
    # emul = GPEmulator.load(experiment, optimizer, loadpath)

    # ================================================================
    # (6) Testing trained GPE at new input points (inference)
    # ================================================================
    y_pred_mean, y_pred_std = emul.predict(X_test)

    r2s = R2Score()(
        tensorize(y_pred_mean),
        tensorize(y_test),
    )
    ise = ISE(tensorize(y_test), tensorize(y_pred_mean), tensorize(y_pred_std))
    print("\nStatistics on test set:")
    print(f"  R2Score = {r2s:.8f}")
    print(f"  ISE = {ise:.2f} %\n")

    if experiment.scaled_data.with_val and not np.isclose(
        r2s, 0.58630764, rtol=1.0e-5
    ):
        log.error("INCORRECT R2Score (with val)")
    if not experiment.scaled_data.with_val and not np.isclose(
        r2s, 0.89883888, rtol=1.0e-5
    ):
        log.error("INCORRECT R2Score")

    # ================================================================
    # (7) Plotting predictions vs observations
    # ================================================================
    height = 9.36111
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 3))

    l = np.argsort(
        y_pred_mean
    )  # let's sort predicted values for a better visualisation

    ci = 2  # ~95% confidence interval

    axis.scatter(
        np.arange(len(l)),
        y_test[l],
        facecolors="none",
        edgecolors="C0",
        label="observed",
    )
    axis.scatter(
        np.arange(len(l)),
        y_pred_mean[l],
        facecolors="C0",
        s=16,
        label="predicted",
    )
    axis.errorbar(
        np.arange(len(l)),
        y_pred_mean[l],
        yerr=ci * y_pred_std[l],
        c="C0",
        ls="none",
        lw=0.5,
        label=f"uncertainty ({ci} STD)",
    )

    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_ylabel(ylabels[int(idx_feature)], fontsize=12)
    axis.set_title(f"R2Score = {r2s:.4f} | ISE = {ise:.2f} %", fontsize=12)
    axis.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
