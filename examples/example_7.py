#!/usr/bin/env python3
#
# 7. Global sensitivity analysis (simulator- and emulator- based)
#
def main():
    # import main libraries
    import numpy as np
    import torch
    from functools import partial

    # enforce reproducibility
    from GPErks.utils.random import set_seed
    from GPErks.constants import DEFAULT_RANDOM_SEED
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

    # analytic solution
    from GPErks.utils.test_functions_gsa import SobolGstar_theoretical_Si
    df_STi_theo, df_Si_theo, df_Sij_theo = SobolGstar_theoretical_Si(a, delta, alpha)
    print(df_STi_theo)
    print(df_Si_theo)
    print(df_Sij_theo)

    # build dataset using the given function
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
     
    # we use Saltelli method from SALib to numerically estimate Sobol' indices
    # (first-order Si or main effects, second-order Sij, and total effects STi);
    from GPErks.perks.gsa import SobolGSA
    gsa = SobolGSA(dataset, n=1024, seed=seed)

    # the example function chosen (SobolGstar) to perform GSA is computationally
    # inexpensive, so we will directly pass it to the estimator method;
    # a total of n x (2d + 2) function (model) evaluations will be performed
    gsa.estimate_Sobol_indices_with_simulator(f)
    gsa.summary()

    # here we repeat the same process by varying the baseline input sample value (n)
    # to demonstrate that Sobol' indices estimates numerically converge to the true,
    # theoretical values as n increases
    import matplotlib.pyplot as plt
    n = list(np.logspace(2, 4, 7))
    # note: best practice would be to use a power of 2 as a value for n; however,
    # here we wanted to test equally log-spaced values, so this will raise a warning

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(2*width, 2*height/3))

    for n_i in n:
        gsa = SobolGSA(dataset, n=int(n_i), seed=seed)
        gsa.estimate_Sobol_indices_with_simulator(f)
        
        for k, axis in enumerate(axes.flat):
            i = k // 4
            j = k % 4
            
            Si_k, Si_k_std = gsa.S1[0, k], gsa.S1_std[0, k]
            axis.errorbar(np.log10(n_i), Si_k, yerr=2*Si_k_std, fmt="o", color="C0")
            axis.axhline(df_Si_theo["Si"][k], c="r", ls="--", lw=0.8)
            
            if j == 0:
                axis.set_ylim([-0.1, 1])
            if i == 1:
                axis.set_xticks([2, 3, 4])
                axis.set_xticklabels([f"$10^{i}$" for i in [2, 3, 4]])
                
    fig.suptitle("Main effect (Si) estimate vs n", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()


    # if the model we have is too computationally expensive and we cannot afford
    # performing n x (2d + 2) evaluations, we can replace the model with an emulator
    # and estimate Sobol' indices using emulator evaluations which are very cheap!

    # define experiment
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
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
    from GPErks.train.emulator import GPEmulator
    from GPErks.train.early_stop import GLEarlyStoppingCriterion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = GPEmulator(experiment, device)
    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)
    emulator.train(optimizer, early_stopping_criterion=esc)

    # check trained emulator accuracy
    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()

    # perk n.4: GPE-based GSA using full posterior distribution samples to derive
    # Sobol' indices distributions
    gsa = SobolGSA(dataset, n=1024, seed=seed)
    gsa.estimate_Sobol_indices_with_emulator(emulator, n_draws=1000)
    # note: we are sampling 1000 times from the emulator posterior distribution
    gsa.summary()
    # note: occasionally we might observe estimated negative values; these are due to
    # numerical errors and typically converge to zero as the number of samples
    # (n) increases

    # we can additionally post-process the obtained Sobol' indices to neglect
    # parameters whose effects are below a certain threshold (default: 1e-2)
    gsa.correct_Sobol_indices()  # use flag threshold=x to change threshold value
    gsa.summary()
    # note: we can see that now only the most relevant indices are appearing,
    # while all the others are set to zero

    # compare emulator derived Sobol' indices predictions vs true, theoretical values
    import matplotlib.gridspec as grsp
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df_ST = pd.DataFrame(data=gsa.ST, columns=gsa.index_i)
    df_S1 = pd.DataFrame(data=gsa.S1, columns=gsa.index_i)
    df_S2 = pd.DataFrame(
        data=gsa.S2,
        columns=["(" + elem[0] + ", " + elem[1] + ")" for elem in gsa.index_ij]
    )

    plt.style.use("seaborn")
    gs = grsp.GridSpec(2, 2)
    fig = plt.figure(figsize=(2*width, height))

    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(ax=ax0, data=df_S1)
    ax0.set_ylim(0, 1)
    ax0.set_title("First-order effect", fontweight="bold", fontsize=12)
    ax0.set_xticklabels(
        ax0.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l0 = list(df_Si_theo["Si"])
    for k, val in enumerate(l0):
        ax0.axhline(val, c="r", lw=1, ls="--")

    ax1 = fig.add_subplot(gs[0, 1])
    sns.boxplot(ax=ax1, data=df_ST)
    ax1.set_ylim(0, 1)
    ax1.set_title("Total effect", fontweight="bold", fontsize=12)
    ax1.set_xticklabels(
        ax1.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l1 = list(df_STi_theo["STi"])
    for k, val in enumerate(l1):
        ax1.axhline(val, c="r", lw=1, ls="--")

    ax2 = fig.add_subplot(gs[1, :])
    sns.boxplot(ax=ax2, data=df_S2)
    ax2.set_ylim(0, 1)
    ax2.set_title("Second-order effect", fontweight="bold", fontsize=12)
    ax2.set_xticklabels(
        ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    l2 = list(df_Sij_theo["Sij"])
    for k, val in enumerate(l2):
        ax2.axhline(val, c="r", lw=1, ls="--")

    fig.tight_layout()
    plt.show()

    # we provide different kinds of plots to visualize Sobol' indices in different ways;
    # before plotting, we recommend running gsa.correct_Sobol_indices() to zero out
    # possible negative values which might break the plotting routines;
    # default plot is the same showed above but without theoretical values (of course)
    # gsa.plot()
    
    gsa.plot_donut()
    gsa.plot_fancy_donut()
    gsa.plot_heatmap()
    gsa.plot_network()

    # note: latest plotted axis/axes can be retrieved for external manipulations via
    axis = gsa.current_axis
    print(type(axis), axis.shape)  # a numpy array
    print(type(axis[0]))  # with matplotlib axis/axes as element/s


if __name__ == '__main__':
    main()