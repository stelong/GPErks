#!/usr/bin/env python3
#
# 7. Global sensitivity analysis (simulator- and emulator- based)
#
def main():
    # import main libraries
    from functools import partial

    import numpy as np

    from GPErks.constants import DEFAULT_RANDOM_SEED

    # enforce reproducibility
    from GPErks.utils.random import set_seed
    seed = DEFAULT_RANDOM_SEED
    set_seed(seed)

    # import test function to run GSA
    from GPErks.utils.test_functions_gsa import SobolGstar
    d = 8  # SobolGstar parameter space is 8-dimensional
    a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
    delta = np.random.rand(d)
    alpha = np.ones_like(a)
    f = partial(SobolGstar, a=a, delta=delta, alpha=alpha)
    l_bounds, u_bounds = d * [0], d * [1]

    # create dataset from function evaluations
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        f,
        d,
        n_train_samples=160,
        n_val_samples=32,
        n_test_samples=80,
        design="lhs",
        seed=seed,
        l_bounds=l_bounds,
        u_bounds=u_bounds
    )
    dataset.summary()

    # analytic solution
    from GPErks.utils.test_functions_gsa import SobolGstar_theoretical_Si
    df_STi, df_Si, _ = SobolGstar_theoretical_Si(a, delta, alpha)
    print(df_STi)
    print(df_Si)

    # we use Saltelli method to numerically estimate Sobol' indices:
    # main effects (Si) and total effects (STi)

    # the example function chosen (SobolGstar) to perform GSA is computationally
    # inexpensive, so we will directly pass it to the estimator method;
    # a total of n x (2d + 2) function (model) evaluations will be performed
    from GPErks.perks.gsa import SobolGSA
    gsa = SobolGSA(dataset, n=1024, seed=seed)
    gsa.estimate_Sobol_indices_with_simulator(f)
    gsa.summary()

    # here we repeat the same process by varying the baseline input sample value (n)
    # to demonstrate that Sobol' indices estimates numerically converge to the true,
    # theoretical values as n increases (note: chosen n value needs to be a power of 2)
    import matplotlib.pyplot as plt
    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(2*width, 2*height/3))
    for i in range(6, 13):
        n_i = 2**i
        gsa = SobolGSA(dataset, n=n_i, seed=seed)
        gsa.estimate_Sobol_indices_with_simulator(f)
        for k, axis in enumerate(axes.flat):
            Si_k = gsa.S1[0, k]
            axis.errorbar(
                n_i,
                Si_k.reshape(-1, 1),
                yerr=np.array([Si_k - gsa.boot.first_order.confidence_interval.low[k], gsa.boot.first_order.confidence_interval.high[k] - Si_k]).reshape(-1, 1),
                fmt="o", color="C0")
            axis.axhline(df_Si["Si"].iloc[k], c="r", ls="--", lw=0.8)
            axis.set_ylim([-0.1, 1])
            axis.set_title(f"X{k+1}")
    fig.suptitle("S1 (main effect) estimate vs N parameter", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.xscale("log")
    plt.show()

    # let's estimate Sobol' indices using an emulator of the SobolGstar function
    
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
        seed=seed
    )

    # train emulator
    import torch

    from GPErks.train.early_stop import GLEarlyStoppingCriterion
    from GPErks.train.emulator import GPEmulator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emulator = GPEmulator(experiment, device)
    # emulator.train_auto()
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)
    emulator.train(optimizer, early_stopping_criterion=esc)
    emulator.hyperparameters()

    # check trained emulator accuracy
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()

    # emulator-based gsa
    from GPErks.perks.gsa import SobolGSA
    gsa = SobolGSA(dataset, n=1024, seed=seed)
    gsa.estimate_Sobol_indices_with_emulator(emulator, n_draws=1000)
    gsa.correct_Sobol_indices(threshold=1e-2)
    gsa.summary()

    # plotting estimated sobol' indices

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 2)
    gsa.plot(axis=axes[0, 0], type="bar", colors="tab10")
    gsa.plot(axis=axes[0, 1], type="box", colors="tab10")
    gsa.plot(axis=axes[1, 0], type="donut", colors="tab10")
    gsa.plot(axis=axes[1, 1], type="heat", colors="rocket_r")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
