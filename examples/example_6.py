#!/usr/bin/env python3
#
# 6. K-fold cross-validation (Part 2 - advanced training)
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


    # load dataset
    from GPErks.serialization.labels import read_labels_from_file
    data_dir = "data/example_4/"
    X_ = np.loadtxt(data_dir + "X.txt", dtype=float)[:50]
    y_ = np.loadtxt(data_dir + "y.txt", dtype=float)[:50]  # suppose we only have few points
    xlabels = read_labels_from_file(data_dir + "xlabels.txt")
    ylabel = read_labels_from_file(data_dir + "ylabel.txt")[0]


    # split dataset in training and testing sets
    from sklearn.model_selection import train_test_split
    X, X_test, y, y_test = train_test_split(
        X_,
        y_,
        test_size=0.5,
        random_state=seed
    )


    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset(
        X,
        y,
        x_labels=xlabels,
        y_label=ylabel
    )


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
        seed=seed,
        learn_noise=True
    )


    # k-fold cross-validation training
    from GPErks.perks.cross_validation import KFoldCrossValidation
    from GPErks.train.early_stop import GLEarlyStoppingCriterion

    device = "cpu"
    devices = [device]
    kfcv = KFoldCrossValidation(experiment, devices, n_splits=5, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(
        max_epochs=1000, alpha=0.1, patience=8
    )
    best_model_dct, best_train_stats_dct, test_scores_dct = kfcv.train(
        optimizer,
        esc,
        leftout_is_val=True
    )


    # resulting mean test scores
    for key in test_scores_dct.keys():
        print(f"Mean test {key} score: {np.mean(test_scores_dct[key]):.4f}")


    # check training stats at each split
    best_epochs = []
    for i, bts in best_train_stats_dct.items():
        bts.plot(with_early_stopping_criterion=True)
        best_epochs.append( bts.best_epoch )

    print( best_epochs )


    # train emulator on all the available points
    from GPErks.train.emulator import GPEmulator
    from GPErks.train.early_stop import NoEarlyStoppingCriterion

    dataset = Dataset(
        X,
        y,
        X_test=X_test,
        y_test=y_test,
        x_labels=xlabels,
        y_label=ylabel
    )

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
    device = "cpu"

    emulator = GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    max_epochs = int( np.mean(best_epochs) )  # making use of cross-validation knowledge
    esc = NoEarlyStoppingCriterion(max_epochs)

    best_model, best_train_stats = emulator.train(
        optimizer,
        esc
    )


    best_train_stats.plot(with_early_stopping_criterion=True)


    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()


if __name__ == '__main__':
    main()