#!/usr/bin/env python3
import os

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from scipy.stats import qmc
from torchmetrics import ExplainedVariance, MeanSquaredError, R2Score

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.perks.diagnostics import Diagnostics
from GPErks.perks.inference import Inference
from GPErks.train.early_stop import PkEarlyStoppingCriterion
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import EveryEpochSnapshottingCriterion
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import currin_exp

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
    # fig, axis = plt.subplots(1, 1)
    # axis.scatter(X_train[:, 0], X_train[:, 1], fc="C0", ec="C0", label="train")
    # axis.scatter(X_val[:, 0], X_val[:, 1], fc="C1", ec="C1", label="val")
    # axis.scatter(X_test[:, 0], X_test[:, 1], fc="C2", ec="C2", label="test")
    # plt.legend()
    # plt.show()

    D = 2
    factor = 10
    f = lambda X: np.array([currin_exp(x) for x in X])
    n_train_samples = factor * D
    n_val_samples = int(factor / 4 * D)
    n_test_samples = factor * D

    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples,
        n_val_samples,
        n_test_samples,
        design="lhs",
        seed=8,
    )

    # ================================================================
    # (3) Training GPE
    # ================================================================
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    input_size = dataset.input_size
    mean_function = LinearMean(input_size=input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=input_size))
    metrics = [ExplainedVariance(), R2Score(), MeanSquaredError()]

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
    D = Diagnostics(emul)
    D.summary()
    D.plot()

    I = Inference(emul)
    I.summary()
    I.plot()
    I.interpolate_2Dgrid(f)  # only works for 2D inputs


if __name__ == "__main__":
    main()
