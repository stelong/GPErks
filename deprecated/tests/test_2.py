#!/usr/bin/env python3
import os

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from torchmetrics import MeanSquaredError, R2Score

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.train.early_stop import PkEarlyStoppingCriterion
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
    # loadpath = sys.argv[1].rstrip("/") + "/"
    # X = np.loadtxt(loadpath + "X.txt", dtype=float)
    # Y = np.loadtxt(loadpath + "Y.txt", dtype=float)

    # # xlabels = read_labels_from_file(loadpath + "xlabels.txt")
    # ylabels = read_labels_from_file(loadpath + "ylabels.txt")
    # # plot_dataset(X, Y, xlabels, ylabels)

    N = 10
    x = np.random.rand(N)
    a, b = -2 * np.pi, 2 * np.pi
    x = (b - a) * x + a
    y = np.sin(x)

    dataset = Dataset(x, y)

    # ================================================================
    # (2) Building example training and validation datasets
    # ================================================================
    # idx_feature = sys.argv[2]
    # print(f"\n{ylabels[int(idx_feature)]} feature selected for emulation.")

    # y = np.copy(Y[:, int(idx_feature)])

    # X_, X_test, y_, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=seed
    # )
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_, y_, test_size=0.2, random_state=seed
    # )

    # ================================================================
    # (3) Training GPE
    # ================================================================
    # savepath = sys.argv[3].rstrip("/") + "/" + idx_feature + "/"
    # Path(savepath).mkdir(parents=True, exist_ok=True)

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

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    # early_stopping_criterion = NoEarlyStoppingCriterion(33)  # TODO: investigate if snapshot is required anyway

    MAX_EPOCHS = 1000
    early_stopping_criterion = PkEarlyStoppingCriterion(
        MAX_EPOCHS, alpha=1.0, patience=8, strip_length=20
    )

    here = os.path.abspath(os.path.dirname(__file__))
    # snapshotting_criterion = NeverSaveSnapshottingCriterion(
    snapshotting_criterion = EveryEpochSnapshottingCriterion(
        here + "/snapshot/test_2/restart_{restart}/",
        "epoch_{epoch}.pth",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emul = GPEmulator(experiment, device)
    emul.train(
        optimizer,
        early_stopping_criterion,
        snapshotting_criterion,
    )

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

    set_seed(888)

    x_test = np.random.rand(N)
    x_test = (b - a) * x_test + a
    y_test = np.sin(x_test)

    xx = np.linspace(a, b, 1000)
    yy_mean, yy_std = emul.predict(xx)

    y_pred_mean, y_pred_std = emul.predict(x_test)

    r2s = R2Score()(
        tensorize(y_pred_mean),
        tensorize(y_test),
    )
    ise = ISE(tensorize(y_test), tensorize(y_pred_mean), tensorize(y_pred_std))
    print("\nStatistics on test set:")
    print(f"  R2Score = {r2s:.8f}")
    print(f"  ISE = {ise:.2f} %\n")

    fig, axis = plt.subplots(1, 1)
    axis.fill_between(
        xx, yy_mean - 2 * yy_std, yy_mean + 2 * yy_std, color="C0", alpha=0.15
    )
    axis.plot(xx, yy_mean, c="C0")
    axis.scatter(x, y, fc="C0", ec="C0")
    axis.scatter(x_test, y_test, fc="C3", ec="C3")
    plt.show()

    # if experiment.scaled_data.with_val and not np.isclose(
    #     r2s, 0.58630764, rtol=1.0e-5
    # ):
    #     log.error("INCORRECT R2Score (with val)")
    # if not experiment.scaled_data.with_val and not np.isclose(
    #     r2s, 0.89883888, rtol=1.0e-5
    # ):
    #     log.error("INCORRECT R2Score")

    # ================================================================
    # (7) Plotting predictions vs observations
    # ================================================================
    # height = 9.36111
    # width = 5.91667
    # fig, axis = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 3))

    # l = np.argsort(
    #     y_pred_mean
    # )  # let's sort predicted values for a better visualisation

    # ci = 2  # ~95% confidence interval

    # axis.scatter(
    #     np.arange(len(l)),
    #     y_test[l],
    #     facecolors="none",
    #     edgecolors="C0",
    #     label="observed",
    # )
    # axis.scatter(
    #     np.arange(len(l)),
    #     y_pred_mean[l],
    #     facecolors="C0",
    #     s=16,
    #     label="predicted",
    # )
    # axis.errorbar(
    #     np.arange(len(l)),
    #     y_pred_mean[l],
    #     yerr=ci * y_pred_std[l],
    #     c="C0",
    #     ls="none",
    #     lw=0.5,
    #     label=f"uncertainty ({ci} STD)",
    # )

    # axis.set_xticks([])
    # axis.set_xticklabels([])
    # axis.set_ylabel(ylabels[int(idx_feature)], fontsize=12)
    # axis.set_title(f"R2Score = {r2s:.4f} | ISE = {ise:.2f} %", fontsize=12)
    # axis.legend(loc="upper left")

    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
