#!/usr/bin/env python3
#
# 2. 2D example (advanced inference, diagnostics)
#
def main():
    # import main libraries
    import torch

    # enforce reproducibility
    from GPErks.utils.random import set_seed
    from GPErks.constants import DEFAULT_RANDOM_SEED
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
    dataset.summary()
    dataset.plot()  # plot (multi-dimensional) input vs scalar output to check for relationships
    dataset.plot_pairwise()  # plot input vs input to check for parameter space coverage

    # choose likelihood
    from gpytorch.likelihoods import GaussianLikelihood
    likelihood = GaussianLikelihood()

    # choose mean function
    from GPErks.gp.mean import LinearMean
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)

    # choose kernel
    from gpytorch.kernels import MaternKernel, ScaleKernel
    kernel = ScaleKernel(MaternKernel(ard_num_dims=dataset.input_size))

    # choose metrics
    from torchmetrics import R2Score
    from GPErks.utils.metrics import IndependentStandardError  # not available from torchmetrics -> we implemented it
    metrics = [IndependentStandardError(), R2Score()]

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

    # check fitted hyperparameters
    emulator.hyperparameters()

    # inference on stored test set
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()
    # bonus function: inference on 2-dimensional grid (only works for R^2->R functions)
    inference.interpolate_2Dgrid(currin_exp)  # can also call with no argument

    # # perk n.2: diagnostics
    # from GPErks.perks.diagnostics import Diagnostics
    # diagnostics = Diagnostics(emulator)
    # diagnostics.summary()
    # diagnostics.plot(errors_type="correlated")
    # diagnostics.plot(errors_type="uncorrelated")
    # diagnostics.plot(errors_type="pivoted")

    # # diagnostics' calculation is based on posterior predicted covariance matrix;
    # # this can be derived by setting flag to True:
    # y_mean, y_std, y_covar = emulator.predict(dataset.X_test, with_covar=True)
    # print(y_covar.shape)

    # import matplotlib.pyplot as plt
    # fig, axis = plt.subplots(1, 1)
    # h = axis.imshow(y_covar)
    # fig.colorbar(h, ax=axis)
    # fig.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
