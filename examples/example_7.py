#!/usr/bin/env python3
#
# 7. Global sensitivity analysis (simulator- and emulator- based)
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


    # import gsa test function
    from GPErks.utils.test_functions_gsa import SobolGstar
    D = 8
    a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
    delta = np.random.rand(D)
    alpha = np.ones_like(a)
    f = lambda X: np.array([SobolGstar(x, a, delta, alpha) for x in X])
    l_bounds, u_bounds = D * [0], D * [1]


    # analytic solution
    from GPErks.utils.test_functions_gsa import SobolGstar_theoretical_Si
    df_STi_theo, df_Si_theo, df_Sij_theo = SobolGstar_theoretical_Si(a, delta, alpha)
    print( df_STi_theo )
    print( df_Si_theo )
    print( df_Sij_theo )


    # build dataset using the given function
    from GPErks.gp.data.dataset import Dataset
    dataset = Dataset.build_from_function(
        f,
        D,
        n_train_samples=160,
        n_val_samples=32,
        n_test_samples=80,
        design="lhs",
        seed=seed,
        l_bounds=l_bounds,
        u_bounds=u_bounds
    )
    dataset.plot()
    dataset.plot_pairwise()
     

    # Saltelli method for Sobol' indexes (Si) estimates
    from GPErks.perks.gsa import SobolGSA
    gsa = SobolGSA(dataset, n=1024, seed=seed)


    # calculate Si using model evaluations
    gsa.estimate_Sobol_indices_with_simulator(f)
    gsa.summary()


    # Si estimates improve as N grows
    import matplotlib.pyplot as plt
    n = list( np.logspace(2, 4, 7) )

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
                
    fig.suptitle("Main effect estimate vs n", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()


    # define experiment
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from torchmetrics import MeanSquaredError, R2Score
    from GPErks.gp.experiment import GPExperiment
    from GPErks.gp.mean import LinearMean
    from GPErks.train.emulator import GPEmulator
    from GPErks.train.early_stop import GLEarlyStoppingCriterion

    likelihood = GaussianLikelihood()
    mean_function = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
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
    esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

    best_model, best_train_stats = emulator.train(optimizer, esc)


    best_train_stats.plot(with_early_stopping_criterion=True)


    from GPErks.perks.inference import Inference
    inference = Inference(emulator)
    inference.summary()
    inference.plot()


    from GPErks.perks.diagnostics import Diagnostics
    diagnostics = Diagnostics(emulator)
    diagnostics.summary()
    diagnostics.plot()


    # estimate Si using the emulator
    gsa = SobolGSA(dataset, n=1024, seed=seed)
    gsa.estimate_Sobol_indices_with_emulator(emulator, n_draws=1000)
    gsa.summary()


    import matplotlib.gridspec as grsp
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df_ST = pd.DataFrame(data=gsa.ST, columns=gsa.index_i)
    df_S1 = pd.DataFrame(data=gsa.S1, columns=gsa.index_i)
    df_S2 = pd.DataFrame(
        data=gsa.S2,
        columns=[
            "(" + elem[0] + ", " + elem[1] + ")" for elem in gsa.index_ij
        ],
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
    trans0 = ax0.get_xaxis_transform()
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
    trans1 = ax1.get_xaxis_transform()
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
    trans2 = ax2.get_xaxis_transform()
    for k, val in enumerate(l2):
        ax2.axhline(val, c="r", lw=1, ls="--")

    fig.tight_layout()
    plt.show()


    gsa.correct_Sobol_indices(threshold=0.05)
    gsa.plot()


    gsa.plot_donut()
    gsa.plot_fancy_donut()
    gsa.plot_heatmap()
    gsa.plot_network()


if __name__ == '__main__':
    main()