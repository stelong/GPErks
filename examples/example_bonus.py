#!/usr/bin/env python3
#
# Bonus. GPE auto-training + GSA using external (from publication) dataset loaded from json file
#
import os

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel

from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.gp.mean import LinearMean
from GPErks.log.logger import get_logger
from GPErks.serialization.path import posix_path
from GPErks.train.emulator import GPEmulator
from GPErks.utils.random import set_seed
from GPErks.perks.gsa import SobolGSA


def main():
    get_logger()
    seed = 8
    set_seed(seed)

    device = "cpu"

    # This new method loads your dataset into a dictionary where keys = features, values = Dataset objects
    # (each Dataset is built to create the experiment that will emulate the corresponding scalar feature (key))
    datasets = Dataset.build_from_file(posix_path(os.getcwd(), "data", "datasets", "Stefano_8p_sham.json"))
    features = list(datasets.keys())
    print(features)  # available features to be emulated

    # # Note: if you want to create a .json file containing your dataset, you can do so like this:
    # X = np.loadtxt(data_dir / "X.txt", dtype=float)
    # Y = np.loadtxt(data_dir / "Y.txt", dtype=float)
    # xlabels = read_labels_from_file(data_dir / "xlabels.txt")
    # ylabels = read_labels_from_file(data_dir / "ylabels.txt")
    # data_dct = {
    #     "X_train": X.tolist(),
    #     "Y_train": Y.tolist(),
    #     # "X_val": X_val.tolist(), # (if available, otherwise can omit this dct key)
    #     # "Y_val": Y_val.tolist(), # (if available, otherwise can omit this dct key)
    #     # "X_test": X_test.tolist(), # (if available, otherwise can omit this dct key)
    #     # "Y_test": Y_test.tolist(), # (if available, otherwise can omit this dct key)
    #     "x_labels": xlabels,  # (if available, otherwise can omit this dct key)
    #     "y_labels": ylabels,  # (if available, otherwise can omit this dct key)
    #     # "l_bounds": a list here (if available, otherwise can omit this dct key)
    #     # "u_bounds": a list here (if available, otherwise can omit this dct key)
    #     "info": "A short description about the dataset"
    # }
    # with open(Path(os.getcwd())/"datasetname.json", "w") as f:
    #     json.dump(data_dct, f, indent=4)
    #
    # # Also note that there is already a utility function that does this for you:
    # # from GPErks.utils.jsonfiles import create_json_dataset_from_arrays

    feature = "EDV"  # we will emulate just one feature as an example

    # GPE auto-training
    print(f"\nEmulating target feature: {feature}")
    dataset = datasets[feature]

    likelihood = GaussianLikelihood()
    mean = LinearMean(degree=1, input_size=dataset.input_size, bias=True)
    covariance = ScaleKernel(MaternKernel(ard_num_dims=dataset.input_size))
    metrics = []

    experiment = GPExperiment(
        dataset,
        likelihood,
        mean,
        covariance,
        metrics=metrics,
        seed=seed
    )

    emulator = GPEmulator(experiment, device)
    emulator.train_auto()  # you could use a more manual approach here with early stopping etc.
    msg = experiment.print_stats()
    print(f"\nFitted emulator hyperparameters:{msg}")

    # GSA
    gsa = SobolGSA(dataset, n=1024, seed=seed)
    # the following method is used to perform GSA whenever a (trained) emulator object is available, also covering
    # the case where it was trained using an externally imported dataset as in this example
    gsa.estimate_Sobol_indices_with_emulator(emulator, n_draws=1000)
    gsa.summary()

    gsa.correct_Sobol_indices(threshold=0.01)
    gsa.plot()
    gsa.plot_donut()
    gsa.plot_fancy_donut()
    gsa.plot_heatmap()
    gsa.plot_network()


if __name__ == "__main__":
    main()
