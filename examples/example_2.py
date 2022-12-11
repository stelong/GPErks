#!/usr/bin/env python3
#
# 2. 2D example (advanced inference, diagnostics)
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
    dataset.plot()
    dataset.plot_pairwise()


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


    # choose training options: device + optimizer
    device = "cpu"
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)


    # train model
    from GPErks.train.emulator import GPEmulator
    emulator = GPEmulator(experiment, device)
    emulator.train(optimizer)


    # inference on stored test set
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()


    # bonus: inference on 2-dimensional grid
    inference.interpolate_2Dgrid(f)


    # perk n.2: diagnostics
    from GPErks.perks.diagnostics import Diagnostics
    diagnostics = Diagnostics(emulator)

    y_mean, y_std, y_covar = emulator.predict(dataset.X_test, with_covar=True)
    print( y_covar.shape )


    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(1, 1)
    h = axis.imshow(y_covar)
    cbar = fig.colorbar(h, ax=axis)
    fig.tight_layout()
    plt.show()


    diagnostics.summary()


    diagnostics.plot(errors_type="correlated")
    diagnostics.plot(errors_type="uncorrelated")
    diagnostics.plot(errors_type="pivoted")


if __name__ == '__main__':
    main()