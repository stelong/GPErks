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
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.serialization.labels import read_labels_from_file
from GPErks.train.early_stop import GLEarlyStoppingCriterion
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.array import tensorize
from GPErks.utils.metrics import IndependentStandardError as ISE
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
        X_train, y_train, X_test, y_test, X_val, y_val, xlabels, ylabel
    )
    dataset.plot()
    dataset.plot_pairwise()

    # ================================================================
    # (3) Training GPE
    # ================================================================
    savepath = sys.argv[3].rstrip("/") + "/" + str(target_label_idx) + "/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    # np.savetxt(savepath + "X_train.txt", X_train, fmt="%.6f")
    # np.savetxt(savepath + "y_train.txt", y_train, fmt="%.6f")
    # np.savetxt(savepath + "X_val.txt", X_val, fmt="%.6f")
    # np.savetxt(savepath + "y_val.txt", y_val, fmt="%.6f")

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
    # experiment = load_experiment_from_config_file(
    #     config_file,
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    # )

    here = os.path.abspath(os.path.dirname(__file__))
    # snapc = NeverSaveSnapshottingCriterion(
    snapc = EveryEpochSnapshottingCriterion(
        here + "/snapshot/test_1/restart_{restart}/",
        "epoch_{epoch}.pth",
    )

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # esc = NoEarlyStoppingCriterion(33)  # TODO: investigate if snapshot is required anyway
    MAX_EPOCHS = 1000
    esc = GLEarlyStoppingCriterion(MAX_EPOCHS, alpha=1.0, patience=8)
    # esc = PkEarlyStoppingCriterion(
    #     MAX_EPOCHS, alpha=1.0, patience=8, strip_length=20
    # )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emul = GPEmulator(experiment, device)
    emul.train(optimizer, esc, snapc)

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
    axis.set_ylabel(dataset.ylabel, fontsize=12)
    axis.set_title(f"R2Score = {r2s:.4f} | ISE = {ise:.2f} %", fontsize=12)
    axis.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
