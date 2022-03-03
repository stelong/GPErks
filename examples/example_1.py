#!/usr/bin/env python3
#
# 1. 1D example (training, inference, posterior sampling)
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
    from GPErks.utils.test_functions import forrester
    f = lambda x: forrester(x)
    D = 1


    # build dataset
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples=10,
        n_test_samples=10,
        design="srs",
        seed=seed,
        l_bounds=[0],
        u_bounds=[1]  # can put None if, as in this case, parameters range in [0, 1]
    )


    # choose likelihood
    from gpytorch.likelihoods import GaussianLikelihood
    likelihood = GaussianLikelihood()


    # choose mean function
    from gpytorch.means import LinearMean
    mean_function = LinearMean(input_size=dataset.input_size)


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
        learn_noise=True  # y = f(x) + e, e ~ N(0, sigma^2I)
    )


    # choose training options: device + optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)


    # train model
    from GPErks.train.emulator import GPEmulator
    emulator = GPEmulator(experiment, device)
    emulator.train(optimizer)


    # inference on stored test set
    x_test = dataset.X_test
    y_test = dataset.y_test

    y_mean, y_std = emulator.predict(x_test)

    for metric in metrics:
        print( metric(
            torch.from_numpy(y_mean), torch.from_numpy(y_test)
            ).item()
        )


    # perk n.1: automatic inference
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()  # can be retrieved from inference.scores_dct
    print( inference.scores_dct )


    # nice plotting
    x_train = dataset.X_train
    y_train = dataset.y_train

    xx = np.linspace(dataset.l_bounds[0], dataset.u_bounds[0], 1000)
    yy_mean, yy_std = emulator.predict(xx)
    yy_true = f(xx)

    import matplotlib.pyplot as plt
    height = 9.36111
    width = 5.91667
    fig, axis = plt.subplots(1, 1, figsize=(4*width/3, height/2))

    axis.plot(xx, yy_true, c="C0", ls="--", label="true function")

    CI = 2
    axis.plot(xx, yy_mean, c="C0", label="predicted mean")
    axis.fill_between(
        xx, yy_mean - CI * yy_std, yy_mean + CI * yy_std, color="C0", alpha=0.15, label="~95% CI"
    )
    axis.scatter(x_train, y_train, fc="C0", ec="C0", label="training data")

    axis.scatter(x_test, y_test, fc="none", ec="C0", label="testing data")

    axis.legend(loc="best")
    fig.tight_layout()
    plt.show()


    # check testing points
    inference.plot()


    # draw samples from the posterior distribution
    y_mean, y_std = emulator.predict(x_test)
    print(y_mean.shape)
    print(y_std.shape)


    y_samples = emulator.sample(x_test, n_draws=5)
    print(y_samples.shape)


    y_samples = emulator.sample(xx, n_draws=5)

    fig, axis = plt.subplots(1, 1, figsize=(4*width/3, height/2))

    for i, ys in enumerate(y_samples):
        axis.plot(xx, ys, lw=0.8, label=f"posterior sample #{i+1}", zorder=1)
        
    axis.plot(xx, yy_mean, c="k", lw=2, ls="--", label="posterior mean", zorder=2)
    axis.scatter(x_train, y_train, fc="k", ec="k", label="training data", zorder=2)

    axis.legend(loc="best")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()