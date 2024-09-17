#!/usr/bin/env python3
#
# 4. Training stats (losses monitoring, early stopping)
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

    # load externally generated dataset
    from GPErks.serialization.path import posix_path
    data_dir = Path(posix_path(os.getcwd(), "examples", "data", "example_4"))
    x = np.loadtxt(data_dir / "X.txt", dtype=float)
    y = np.loadtxt(data_dir / "y.txt", dtype=float)

    # split original dataset in training, validation and testing sets
    from sklearn.model_selection import train_test_split
    x_, x_test, y_, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_,
        y_,
        test_size=0.2,
        random_state=seed
    )

    # load input parameters and output feature names
    from GPErks.serialization.labels import read_labels_from_file
    x_labels = read_labels_from_file(data_dir / "xlabels.txt")
    y_label = read_labels_from_file(data_dir / "ylabel.txt")[0]  # there is only one element in the list

    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset(
        x_train,
        y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        x_labels=x_labels,
        y_label=y_label,
        name="CanopyReflectance",
        descr="A reflectance model for the homogeneous plant canopy and its inversion (doi.org/10.1016/0034-4257(89)90015-1)"
    )
    dataset.summary()
    dataset.plot()
    dataset.plot_pairwise()

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
        n_restarts=1,
        metrics=metrics,
        seed=seed,  # reproducible training
    )

    # train model
    from GPErks.train.emulator import GPEmulator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    emulator = GPEmulator(experiment, device)
    # when training, we can actually output both the best model state (a dict) and some training statistics (an object)
    best_model, best_train_stats = emulator.train(optimizer)
    print(list(best_model.keys()))
    # training statistics can be used to examine training and validation losses
    best_train_stats.plot()

    # usual inference on test set
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()

    # early stopping - stop model training at a given epoch
    # by default, training is performed for a fixed number of iterations (n_epochs = 100 by default);
    # for this reason, the best epoch (i.e., epoch that achieved the lowest training or validation loss) would be 100
    print(best_train_stats.best_epoch == 100)
    # we can plot the criterion used for stopping by setting flag to True
    best_train_stats.plot(with_early_stopping_criterion=True)
    # since no criterion was used, the plot shows a flat line to the default value of zero

    # we can run the training to a different, fixed number of epochs
    from GPErks.train.early_stop import NoEarlyStoppingCriterion
    max_epochs = 50
    esc = NoEarlyStoppingCriterion(max_epochs)
    _, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )
    inference = Inference(emulator)
    inference.summary()
    print(best_train_stats.best_epoch == max_epochs)
    best_train_stats.plot(with_early_stopping_criterion=True)

    # we have implemented different criteria, but you can implement your own stopping criterion
    # by following the same class structure as the ones imported here
    from GPErks.train.early_stop import (
        GLEarlyStoppingCriterion,
        PQEarlyStoppingCriterion,
        UPEarlyStoppingCriterion,
    )
    max_epochs = 500
    esc = GLEarlyStoppingCriterion(
            max_epochs, alpha=0.1, patience=8
    )
    # esc = PQEarlyStoppingCriterion(
    #     max_epochs, alpha=1, patience=8, strip_length=5
    # )
    # esc = UPEarlyStoppingCriterion(
    #     max_epochs, strip_length=5, successive_strips=4
    # )
    _, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )
    inference = Inference(emulator)
    inference.summary()
    print(best_train_stats.best_epoch == max_epochs)  # False: early stopping did its job
    best_train_stats.plot(with_early_stopping_criterion=True)
    # criterion is no more a flat line but follows a specific trend; stopping took place when this crossed
    # the value of 0.1 (alpha parameter) for more than 8 epochs (patience parameter)

    # early stopping is based on evaluating the loss on a validation set;
    # in case we don't have a validation set, we can still come up with some criterion that stops the training
    # according to convergence of training loss to a plateau
    from GPErks.train.early_stop import PkEarlyStoppingCriterion
    esc = PkEarlyStoppingCriterion(
        max_epochs, alpha=0.01, patience=8, strip_length=20
    )
    # to make this work, we need to hide the fact that we created a dataset providing a validation set;
    # let's pretend we don't have a validation set using this trick
    emulator.scaled_data.with_val = False
    # (we could have achieved the same result by creating a new dataset with no validation set,
    # a new experiment and a new emulator object)

    _, best_train_stats = emulator.train(
        optimizer,
        early_stopping_criterion=esc
    )
    inference = Inference(emulator)
    inference.summary()
    print(best_train_stats.best_epoch == max_epochs)
    best_train_stats.plot(with_early_stopping_criterion=True)
    # we were able to stop training very early, which is good since it prevents overfitting;
    # however, we advise not to rely solely on training loss and to always use a validation set when training


if __name__ == '__main__':
    main()
