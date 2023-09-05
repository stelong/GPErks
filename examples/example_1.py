#!/usr/bin/env python3
#
# 1. 1D example (training, inference, posterior sampling)
#
def main():
    # import main libraries
    import numpy as np
    import torch

    # enforce reproducibility
    from GPErks.utils.random import set_seed
    from GPErks.constants import DEFAULT_RANDOM_SEED
    seed = DEFAULT_RANDOM_SEED
    set_seed(seed)  # reproducible sampling

    # function to learn (1D input in this case;
    # normally, an R^d -> R function which is computationally expensive to run)
    from GPErks.utils.test_functions import forrester
    d = 1  # forrester input is 1D

    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        forrester,
        d,
        n_train_samples=10,
        n_test_samples=10,
        design="srs",
        seed=seed,
        l_bounds=[0],  # for d-dimensional input, l_bounds and u_bounds must be lists of length = d
        u_bounds=[1],  # could omit l_bounds and u_bounds if, as in this case, input parameters live in [0, 1]
    )
    # we can get a coincise summary of the dataset like this:
    dataset.summary()

    # choose likelihood
    from gpytorch.likelihoods import GaussianLikelihood
    likelihood = GaussianLikelihood()

    # choose mean function
    from GPErks.gp.mean import LinearMean
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)

    # choose covariance function (kernel)
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
        seed=seed,  # reproducible training
    )

    # choose training options: device + optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    # train model
    from GPErks.train.emulator import GPEmulator
    emulator = GPEmulator(experiment, device)
    emulator.train(optimizer)

    # check fitted hyperparameters;
    # by default, after training, the best emulator stored in the emulator instance is the one that achieved
    # the lowest training loss (or validation loss if we provided a validation dataset) across the different
    # restarts we have run
    emulator.hyperparameters()

    # inference on stored test set
    from GPErks.utils.array import tensorize
    from GPErks.utils.metrics import get_metric_name
    x_test = dataset.X_test
    y_test = dataset.y_test
    y_mean, y_std = emulator.predict(x_test)
    print(y_mean.shape, y_std.shape)  # 1D arrays of length = dataset.sample_size
    for metric in metrics:
        name = get_metric_name(metric)
        score = metric(tensorize(y_mean), tensorize(y_test)).item()
        print(f"{name} = {score:.4f}")

    # perk n.1: automatic inference
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    print(inference.scores_dct)  # to retrieve raw values
    inference.plot()  # to check performance on test points

    # plotting
    import matplotlib.pyplot as plt
    x_train = dataset.X_train
    y_train = dataset.y_train

    xx = np.linspace(dataset.l_bounds[0], dataset.u_bounds[0], 1000)
    yy_mean, yy_std = emulator.predict(xx)
    yy_true = forrester(xx)

    height = 9.36111
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(4*width/3, height/2))
    axis.plot(xx, yy_true, c="C0", ls="--", label="true function")
    ci = 2
    axis.plot(xx, yy_mean, c="C0", label="predicted mean")
    axis.fill_between(
        xx, yy_mean - ci * yy_std, yy_mean + ci * yy_std, color="C0", alpha=0.15, label="~95% CI"
    )
    axis.scatter(x_train, y_train, fc="C0", ec="C0", label="training data")
    axis.scatter(x_test, y_test, fc="none", ec="C0", label="testing data")
    axis.legend(loc="best")
    fig.tight_layout()
    plt.show()

    # draw samples from the posterior distribution
    y_samples = emulator.sample(xx, n_draws=5)
    print(y_samples.shape)  # samples are stored row-wise

    fig, axis = plt.subplots(1, 1, figsize=(4*width/3, height/2))
    for i, ys in enumerate(y_samples):
        axis.plot(xx, ys, lw=0.8, label=f"posterior sample #{i+1}", zorder=1)
    axis.plot(xx, yy_mean, c="k", label="posterior mean", zorder=2)
    axis.fill_between(
        xx, yy_mean - ci * yy_std, yy_mean + ci * yy_std, color="k", alpha=0.15, label="~95% CI"
    )
    axis.scatter(x_train, y_train, fc="k", ec="k", label="training data", zorder=2)
    axis.legend(loc="best")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
