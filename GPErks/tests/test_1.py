#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from sklearn.model_selection import train_test_split

from GPErks.data import ScaledData
from GPErks.gpe import GPEmul, LEARNING_RATE
from GPErks.models.models import ExactGPModel, LinearMean
from GPErks.utils.design import read_labels
from GPErks.utils.metrics import R2Score
from GPErks.utils.plotting import plot_dataset
from GPErks.utils.preprocessing import UnitCubeScaler, StandardScaler
from GPErks.utils.tensor import tensorize

SEED = 8
KERNEL_DCT = {"Matern": MaternKernel, "RBF": RBFKernel}
KERNEL = "RBF"
LOG_TRANSFORM = False


def main():
    # ================================================================
    # (0) Making the code reproducible
    # ================================================================
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # (1) Loading and visualising dataset
    # ================================================================
    loadpath = sys.argv[1].rstrip("/") + "/"
    X = np.loadtxt(loadpath + "X.txt", dtype=float)
    Y = np.loadtxt(loadpath + "Y.txt", dtype=float)

    xlabels = read_labels(loadpath + "xlabels.txt")
    ylabels = read_labels(loadpath + "ylabels.txt")
    plot_dataset(X, Y, xlabels, ylabels)

    # ================================================================
    # (2) Building example training and validation datasets
    # ================================================================
    idx_feature = sys.argv[2]
    print(f"\n{ylabels[int(idx_feature)]} feature selected for emulation.")

    y = np.copy(Y[:, int(idx_feature)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
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

    X_scaler = UnitCubeScaler()
    y_scaler = StandardScaler(log_transform=LOG_TRANSFORM)
    train_scaled_data = ScaledData(
        # X_train, y_train, X_scaler, y_scaler, X_val, y_val  # TODO
        X_train, y_train, X_scaler, y_scaler,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    linear_model = LinearMean(
        input_size=train_scaled_data.input_size, data_mean=0.0
    )
    kernel = ScaleKernel(
        KERNEL_DCT[KERNEL](ard_num_dims=train_scaled_data.input_size)
    )

    model = ExactGPModel(
        train_scaled_data.X_train,
        train_scaled_data.y_train,
        likelihood,
        linear_model,
        kernel,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    emul = GPEmul(train_scaled_data, model, optimizer)
    emul.train(save_losses=True, savepath=savepath)

    # ================================================================
    # (4) Saving trained GPE
    # ================================================================
    emul.save()

    # ================================================================
    # (5) Loading already trained GPE
    # ================================================================
    # NOTE: you need exactly the same training dataset used in (3)
    # ================================================================
    loadpath = savepath
    emul = GPEmul.load(train_scaled_data, model, optimizer, loadpath)

    # ================================================================
    # (6) Testing trained GPE at new input points (inference)
    # ================================================================
    # NOTE: we will use the validation dataset used in (3) as an example
    # ================================================================
    X_test = X_val
    y_test = y_val

    y_pred_mean, y_pred_std = emul.predict(X_test)
    r2s = R2Score(tensorize(y_test), tensorize(y_pred_mean))
    print(f"\nAccuracy on testing dataset: R2Score = {r2s:.6f}")

    # ================================================================
    # (7) Plotting predictions vs observations
    # ================================================================
    height = 9.36111
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 3))

    l = np.argsort(
        y_pred_mean
    )  # let's sort predicted values for a better visualisation
    ci = 3

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
    axis.set_title(f"R2Score = {r2s:.6f}", fontsize=12)
    axis.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
