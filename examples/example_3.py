#!/usr/bin/env python3
#
# 3. I/O handling (reproducible experiments, snapshotting)
#
def main():
    import numpy as np
    import torch


    # set logger and enforce reproducibility
    from GPErks.log.logger import get_logger
    from GPErks.utils.random import set_seed
    log = get_logger()
    seed = 8
    set_seed(seed)  # reproducible sampling


    # function to learn (normally a high-dimensional, expensive deterministic model)
    from GPErks.utils.test_functions import currin_exp
    f = lambda X: np.array([currin_exp(x) for x in X])
    D = 2


    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples=20,
        n_test_samples=25,
        design="lhs",
        seed=seed,
    )


    # choose likelihood
    from gpytorch.likelihoods import GaussianLikelihood
    likelihood = GaussianLikelihood()


    # choose mean function
    from GPErks.gp.mean import LinearMean
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)


    # choose kernel
    from gpytorch.kernels import RBFKernel, ScaleKernel
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))


    # choose metrics
    from torchmetrics import MeanSquaredError, R2Score
    metrics = [MeanSquaredError(), R2Score()]


    # define experiment
    from GPErks.gp.experiment import GPExperiment
    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed  # reproducible training
    )


    # dump experiment in config file
    config_file = "./example_3.ini"
    experiment.save_to_config_file(config_file)


    # load experiment from config file
    del experiment

    from GPErks.gp.experiment import load_experiment_from_config_file
    experiment = load_experiment_from_config_file(
        config_file,
        dataset  # notice that we still need to provide the dataset used!
    )


    # choose training options: device + optimizer
    device = "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)


    # train model
    from GPErks.train.emulator import GPEmulator
    emulator = GPEmulator(experiment, device)


    # snapshotting
    from GPErks.serialization.path import posix_path
    from GPErks.train.snapshot import (
        EveryEpochSnapshottingCriterion,
        EveryNEpochsSnapshottingCriterion,
        NeverSaveSnapshottingCriterion
    )
    import os

    snapshot_dir = posix_path(os.getcwd(), "snapshot", "example_3")
    train_restart_template = "restart_{restart}"
    train_epoch_template = "epoch_{epoch}.pth"

    snapshot_file = train_epoch_template
    snpc = EveryEpochSnapshottingCriterion(
        posix_path(snapshot_dir, train_restart_template),
        snapshot_file
    )

    # training
    emulator.train(optimizer, snapshotting_criterion=snpc)


    # inference on stored test set
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()


    # loading emulator
    best_model_file = posix_path(
        snapshot_dir,
        "best_model.pth"
    )
    best_model_state = torch.load(best_model_file, map_location=torch.device(device))

    emulator1 = GPEmulator(experiment, device)
    emulator1.model.load_state_dict(best_model_state)


    inference = Inference(emulator1)
    inference.summary()


if __name__ == '__main__':
    main()