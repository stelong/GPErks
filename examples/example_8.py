#!/usr/bin/env python3
#
# 8. Showcasing new functionalities: automatic emulator hyperparameters' fit
# + LinearMean module with higher than first degree polynomials + mean module fit inspection
#
import numpy

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from torchmetrics import MeanSquaredError, R2Score

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.gp.mean import LinearMean
from GPErks.log.logger import get_logger
from GPErks.perks.inference import Inference
from GPErks.plot.mean import inspect_mean_module
from GPErks.train.emulator import GPEmulator
from GPErks.utils.random import set_seed
from GPErks.utils.test_functions import currin_exp


def main():
    # set logger and enforce reproducibility
    log = get_logger()
    seed = 8
    set_seed(seed)  # reproducible sampling

    # function to learn (2D -> 1D)
    f = lambda X: numpy.array([currin_exp(x) for x in X])
    D = 2

    # build dataset
    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples=20,
        n_test_samples=25,
        design="lhs",
        seed=seed,
    )

    # choose likelihood
    likelihood = GaussianLikelihood()

    # choose mean function
    # NOTE: can now choose higher than first degree polynomials!
    # However, most of the time we can get away with degree=1, I have coded this because apparently it is not available
    # anywhere and some people asked to have this functionality added so here you go!
    mean = LinearMean(degree=2, input_size=dataset.input_size, bias=True)

    # choose covariance function
    covar = ScaleKernel(MaternKernel(ard_num_dims=dataset.input_size))

    # choose metrics
    metrics = [MeanSquaredError(), R2Score()]

    # define experiment + device
    experiment = GPExperiment(
        dataset,
        likelihood,
        mean,
        covar,
        metrics=metrics,
        seed=seed
    )
    device = "cpu"

    # emulator training
    # NOTE: can now fit the hyperparameters automatically!
    # However, for small training datasets I would still recommend training in the old-fashioned way,
    # I know it's not a trivial task, but it is more reliable than the automatic training when we have few points.
    emulator = GPEmulator(experiment, device)
    emulator.train_auto()

    # inference on stored test set
    inference = Inference(emulator)
    inference.summary()
    inference.plot()

    # NOTE: can now inspect the linear mean module fit!
    inspect_mean_module(emulator)


if __name__ == '__main__':
    main()
