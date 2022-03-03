#!/usr/bin/env python3
#
# 5. K-fold cross-validation (Part 1 - accuracy)
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
    X_ = np.loadtxt(data_dir + "X.txt", dtype=float)[:80]
    y_ = np.loadtxt(data_dir + "y.txt", dtype=float)[:80]
    xlabels = read_labels_from_file(data_dir + "xlabels.txt")
    ylabel = read_labels_from_file(data_dir + "ylabel.txt")[0]


    # split dataset in training and validation sets
    from sklearn.model_selection import train_test_split
    X, X_val, y, y_val = train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=seed
    )


    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset(
        X,
        y,
        X_val=X_val,
        y_val=y_val,
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
    devices = [device]  # a list of devices
    kfcv = KFoldCrossValidation(experiment, devices, n_splits=5, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)
    best_model_dct, best_train_stats_dct, test_scores_dct = kfcv.train(optimizer, esc)


    # check mean test scores
    for key in test_scores_dct.keys():
        print(f"Mean test {key} score: {np.mean(test_scores_dct[key]):.4f}")


    # check training stats at each split
    for i, bts in best_train_stats_dct.items():
        bts.plot(with_early_stopping_criterion=True)


if __name__ == '__main__':
    main()