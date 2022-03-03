#!/usr/bin/env python3
#
# 4. Training stats (losses monitoring, early stopping)
#
def main():
    import numpy as np
    import torch


    # set logger and enforce reproducibility
    from GPErks.log.logger import get_logger
    from GPErks.utils.random import set_seed
    log = get_logger()
    seed = 8
    set_seed(seed)


    # load externally generated dataset
    data_dir = "data/example_4/"
    X = np.loadtxt(data_dir + "X.txt", dtype=float)
    y = np.loadtxt(data_dir + "y.txt", dtype=float)


    # split original dataset in training, validation and testing sets
    from sklearn.model_selection import train_test_split
    X_, X_test, y_, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=seed
    )


    # load input parameters and output feature names
    from GPErks.serialization.labels import read_labels_from_file
    xlabels = read_labels_from_file(data_dir + "xlabels.txt")
    ylabel = read_labels_from_file(data_dir + "ylabel.txt")[0]  # only element of the list
    print( xlabels )
    print( ylabel )


    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        x_labels=xlabels,
        y_label=ylabel
    )
    dataset.plot()
    dataset.plot_pairwise()


    # define experiment
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import LinearMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from torchmetrics import MeanSquaredError, R2Score
    from GPErks.gp.experiment import GPExperiment

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(input_size=dataset.input_size)
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
    metrics = [MeanSquaredError(), R2Score()]

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,  # reproducible training
        learn_noise=True
    )


    # train model
    from GPErks.train.emulator import GPEmulator
    device = "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    emulator = GPEmulator(experiment, device)
    best_model, best_train_stats = emulator.train(optimizer)


    # examining train and val losses
    best_train_stats.plot()


    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()


    # early stopping criteria
    print( best_train_stats.best_epoch )
    best_train_stats.plot(with_early_stopping_criterion=True)


    from GPErks.train.early_stop import NoEarlyStoppingCriterion
    max_epochs = 50
    esc = NoEarlyStoppingCriterion(max_epochs)

    best_model, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )

    best_train_stats.plot(with_early_stopping_criterion=True)


    from GPErks.train.early_stop import (
        GLEarlyStoppingCriterion,
        PQEarlyStoppingCriterion,
        UPEarlyStoppingCriterion,
    )
    max_epochs = 1000

    esc = GLEarlyStoppingCriterion(
            max_epochs, alpha=0.1, patience=8
    )
    # esc = PQEarlyStoppingCriterion(
    #     max_epochs, alpha=1, patience=8, strip_length=5
    # )
    # esc = UPEarlyStoppingCriterion(
    #     max_epochs, strip_length=5, successive_strips=4
    # )

    best_model, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )


    best_train_stats.plot(with_early_stopping_criterion=True)


    from GPErks.train.early_stop import PkEarlyStoppingCriterion
    esc = PkEarlyStoppingCriterion(
        max_epochs, alpha=0.01, patience=8, strip_length=20
    )

    emulator.scaled_data.with_val = False  # let's pretend we don't have a val set

    best_model, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )


    best_train_stats.plot(with_early_stopping_criterion=True)


if __name__ == '__main__':
    main()