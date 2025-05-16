#
# 5. K-fold cross-validation (Part 1 - accuracy)
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
    x_ = np.loadtxt(data_dir / "X.txt", dtype=float)[
        :80
    ]  # to run this example quickly (5-fold CV) we load less points
    y_ = np.loadtxt(data_dir / "y.txt", dtype=float)[:80]
    xlabels = read_labels_from_file(data_dir / "xlabels.txt")
    ylabel = read_labels_from_file(data_dir / "ylabel.txt")[0]

    # split dataset in training and validation sets
    from sklearn.model_selection import train_test_split

    x, x_val, y, y_val = train_test_split(x_, y_, test_size=0.2, random_state=seed)

    # build dataset
    from GPErks.gp.data.dataset import Dataset

    dataset = Dataset(
        x,
        y,
        X_val=x_val,
        y_val=y_val,
        x_labels=xlabels,
        y_label=ylabel,
        name="CanopyReflectance",
        descr="A reflectance model for the homogeneous plant canopy and its inversion (doi.org/10.1016/0034-4257(89)90015-1)",
    )
    dataset.summary()

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
        learn_noise=True,
    )

    # perk n.3:  K-fold cross-validation training;
    # when providing a validation set as in this example, this will be used to stop training
    # at the right moment for EVERY split, while the accuracy of the split-specific emulator
    # will be obtained by testing against the respective split-specific left-out part of
    # the training dataset
    from GPErks.perks.cross_validation import KFoldCrossValidation
    from GPErks.train.early_stop import GLEarlyStoppingCriterion

    device = "cpu"
    devices = [device]  # a list of devices, in this case we are passing just one
    kfcv = KFoldCrossValidation(experiment, devices, n_splits=5, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)
    # similarly to simple training, when doing cross-validation we can again output both
    # the best model state (a dict) and its training statistics (an object); since we have
    # different folds, these will be dictionaries with fold numbers used as keys
    best_model_dct, best_train_stats_dct = kfcv.train(
        optimizer,
        early_stopping_criterion=esc,
    )
    print(
        list(best_model_dct.keys())
    )  # splits might not be ordered if K-fold CV was run using multiple cores

    # check cv split test scores
    kfcv.summary()

    # check training stats at each split
    for _, bts in best_train_stats_dct.items():
        bts.plot(with_early_stopping_criterion=True)

    # best-split emulator (according to highest R2-score) is automatically available within kfcv instance
    from GPErks.perks.inference import Inference

    inference = Inference(kfcv.emulator)
    inference.summary()

    # check best-split emulator fitted hyperparameters
    kfcv.emulator.hyperparameters()

    # note the size differences between train and test sets here compared to original dataset train and val sets
    kfcv.emulator.experiment.dataset.summary()


if __name__ == "__main__":
    main()
