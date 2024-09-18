#!/usr/bin/env python3
#
# 6. K-fold cross-validation (Part 2 - advanced training)
#
def main():
    # import main libraries
    import os
    from pathlib import Path

    import numpy as np
    import torch

    from GPErks.constants import DEFAULT_RANDOM_SEED

    # enforce reproducibility
    from GPErks.utils.random import set_seed
    seed = DEFAULT_RANDOM_SEED
    set_seed(seed)

    # load dataset
    from GPErks.serialization.labels import read_labels_from_file
    data_dir = Path(os.getcwd()) / "examples" / "data" / "example_4"
    x_ = np.loadtxt(data_dir / "X.txt", dtype=float)[:50]  # suppose that we only have a very small amount of data points
    y_ = np.loadtxt(data_dir / "y.txt", dtype=float)[:50]
    xlabels = read_labels_from_file(data_dir / "xlabels.txt")
    ylabel = read_labels_from_file(data_dir / "ylabel.txt")[0]

    # split dataset in training and testing sets
    from sklearn.model_selection import train_test_split
    x, x_test, y, y_test = train_test_split(
        x_,
        y_,
        test_size=0.5,  # training dataset is very small! (on purpose)
        random_state=seed
    )

    # build dataset with no validation and no testing sets
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset(
        x,
        y,
        x_labels=xlabels,
        y_label=ylabel
    )

    # define experiment
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from torchmetrics import MeanSquaredError, R2Score

    from GPErks.gp.experiment import GPExperiment
    from GPErks.gp.mean import LinearMean

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
    kernel = ScaleKernel(MaternKernel(ard_num_dims=dataset.input_size))
    metrics = [MeanSquaredError(), R2Score()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,
        learn_noise=True
    )

    # K-fold cross-validation training
    from GPErks.perks.cross_validation import KFoldCrossValidation
    from GPErks.train.early_stop import GLEarlyStoppingCriterion

    device = "cpu"
    devices = [device]
    kfcv = KFoldCrossValidation(experiment, devices, n_splits=5)
    # note: if we do not specify 'max_workers' above, the K-fold CV will
    # automatically run in multiprocessing using all available cores

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)
    _, best_train_stats_dct = kfcv.train(
        optimizer,
        early_stopping_criterion=esc,
        leftout_is_val=True
    )
    # using this new flag 'leftout_is_val=True', we can utilize the respective leftout
    # set of each split to stop the training using a validation loss-based early
    # stopping criterion

    # because of the used flag, the summary metrics below are not to be considered as
    # reliable scores, because we are evaluating the accuracy against the leftout set
    # used both as a validation and as a testing set
    kfcv.summary()

    # check training stats at each split
    best_epochs = []
    for _, bts in best_train_stats_dct.items():
        bts.plot(with_early_stopping_criterion=True)
        best_epochs.append( bts.best_epoch )

    # nevertheless, what we were really interested in was getting the stopping epoch
    # reached at each split of the cross-validation
    print(best_epochs)

    # train emulator on all the available points
    from GPErks.train.early_stop import NoEarlyStoppingCriterion
    from GPErks.train.emulator import GPEmulator

    # build a new dataset now including a testing set
    del dataset
    dataset = Dataset(
        x,
        y,
        X_test=x_test,
        y_test=y_test,
        x_labels=xlabels,
        y_label=ylabel
    )

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
    kernel = ScaleKernel(MaternKernel(ard_num_dims=dataset.input_size))
    metrics = [MeanSquaredError(), R2Score()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
    )
    device = "cpu"
    emulator = GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    # making use of knowledge coming from the performed CV,
    # we run the training for an exact number of epochs
    max_epochs = int( np.mean(best_epochs) )  
    esc = NoEarlyStoppingCriterion(max_epochs)

    _, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )

    best_train_stats.plot(with_early_stopping_criterion=True)

    # note: we now support also ISE2 as a metric!
    from GPErks.utils.metrics import IndependentStandardError
    emulator.experiment.metrics = [IndependentStandardError(), R2Score()]

    # we are achieving a very high R2Score using only 25 points and without a validation set, magic!
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()


if __name__ == '__main__':
    main()
