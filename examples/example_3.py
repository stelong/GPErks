#!/usr/bin/env python3
#
# 3. I/O handling (reproducible experiments, snapshotting)
#
def main():
    # import main libraries
    import os
    import torch

    # set logger and enforce reproducibility
    from GPErks.log.logger import get_logger
    from GPErks.utils.random import set_seed
    from GPErks.constants import DEFAULT_RANDOM_SEED
    get_logger()
    seed = DEFAULT_RANDOM_SEED
    set_seed(seed)  # reproducible sampling

    # function to learn (2D input)
    from GPErks.utils.test_functions import currin_exp
    d = 2  # currin_exp input is 2D

    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        currin_exp,
        d,
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
    from gpytorch.kernels import MaternKernel, ScaleKernel
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dataset.input_size))

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
        seed=seed,  # reproducible training
    )

    # dump experiment to config file;
    # this will allow reproducing the exact, same experimental setup in future contexts
    config_file = "./example_3.ini"
    experiment.save_to_config_file(config_file)

    # load experiment from config file
    del experiment  # let's delete the original experiment before being able to re-create it from file

    from GPErks.gp.experiment import load_experiment_from_config_file
    experiment = load_experiment_from_config_file(
        config_file,
        dataset  # data is not saved in config file to save memory, so we still need to pass the dataset used!
    )

    # choose training options: device + optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    # train model
    from GPErks.train.emulator import GPEmulator
    emulator = GPEmulator(experiment, device)

    # snapshotting - save model state at a given epoch(s)
    from GPErks.serialization.path import posix_path
    from GPErks.train.snapshot import (
        EveryEpochSnapshottingCriterion,
        EveryNEpochsSnapshottingCriterion,
    )
    snapshot_dir = posix_path(os.getcwd(), "snapshot", "example_3")  # provide folder where to save model instance(s)
    train_restart_template = "restart_{restart}"  # provide template for sub-folder name
    train_epoch_template = "epoch_{epoch}.pth"  # provide template for file name

    snapshot_file = train_epoch_template
    snpc = EveryEpochSnapshottingCriterion(
        posix_path(snapshot_dir, train_restart_template),
        snapshot_file,
    )

    # training
    emulator.train(optimizer, snapshotting_criterion=snpc)

    # by default, after training, the best emulator stored in the emulator instance is the one that achieved
    # the lowest training loss (or validation loss if we provided a validation dataset) across the different
    # restarts we have run; this emulator is the one used, for example, to perform inference on test set
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()

    # loading emulator;
    # a symbolic link to the best emulator across the different restarts run is created at the end of the training
    # process, this can be accessed under the name of "best_model.pth":
    best_model_file = posix_path(
        snapshot_dir,
        "best_model.pth"
    )
    # load the best model fitted hyperparameters
    best_model_state = torch.load(best_model_file, map_location=torch.device(device))

    emulator1 = GPEmulator(experiment, device)  # create an emulator instance
    emulator1.model.load_state_dict(best_model_state)  # update its hyperparameters with the loaded state

    inference = Inference(emulator1)
    inference.summary()  # sanity check: metrics' values should be the same as the one obtained soon after training


if __name__ == '__main__':
    main()
