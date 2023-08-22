from configparser import ConfigParser
from typing import List, Optional

import gpytorch
import torch
import torchmetrics
from gpytorch.kernels import ScaleKernel

from GPErks.constants import DEFAULT_EXP_N_RESTARTS
from GPErks.gp.data.data_scaler import StandardScaler, UnitCubeScaler
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.data.scaled_data import ScaledData
from GPErks.gp.model import ExactGPModel
from GPErks.log.logger import get_logger
from GPErks.serialization.config import get_repeatable_section, read_config
from GPErks.serialization.runtime import build_instance, dump_instance
from GPErks.utils.random import set_seed

log = get_logger()


class GPExperiment:
    """
    Container object for all emulator properties and training setup.

    ...

    Attributes
    ----------
    dataset: GPErks.gp.data.dataset.Dataset
        Container object handling data and all their associated properties.
    scaled_data: GPErks.gp.data.scaled_data.ScaledData
        Container object handling default pre- and post-processing transformers (input->UnitCubeScaler,
        output -> StandardScaler) and data transformed through them.
    n_restarts: int
        Number of times we restart the training process from a different point in the hyperparameter space.
    seed: int, optional
        Random seed for reproducibility.
    metrics: list of torchmetrics.Metric, optional
        List of regression metrics to be evaluated against validation and/or testing dataset.
    learn_noise: bool
        Whether to fit likelihood noise (True) of shrink and fix it to a very small value (False, not recommended).
    model: GPErks.gp.model.ExactGPModel
        GPyTorch Gaussian process emulator model implementing a forward method.

    Parameters
    ----------
    likelihood: gpytorch.likelihoods.Likelihood
        Gaussian process emulator likelihood function.
    mean_module: gpytorch.means.Mean
        Gaussian process emulator mean function.
    covar_module: gpytorch.kernels.Kernel
        Gaussian process emulator covariance function.

    Methods
    -------
    load_model()
    print_stats()
    save_to_config_file()

    Examples
    --------

   """

    def __init__(
        self,
        dataset: Dataset,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        *,
        n_restarts: int = DEFAULT_EXP_N_RESTARTS,
        seed: Optional[int] = None,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        learn_noise: bool = True,
    ):
        set_seed(seed)  # set immediately, for reproducible initialization
        self.seed: Optional[int] = seed

        self.dataset = dataset
        # scale data by default
        self.scaled_data: ScaledData = ScaledData(
            dataset,
            UnitCubeScaler(),
            StandardScaler(),
        )

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module

        self.n_restarts: int = n_restarts
        self.metrics: List[torchmetrics.Metric] = metrics or []
        self.learn_noise: bool = learn_noise

        self.model: ExactGPModel = ExactGPModel(
            self.scaled_data.X_train,
            self.scaled_data.y_train,
            self.likelihood,
            self.mean_module,
            self.covar_module,
        )

    def load_model(
        self,
        model_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        log.info("Loaded model with hyperparameters:")
        log.info(self.print_stats())

    def print_stats(self):
        torch.set_printoptions(sci_mode=False)
        msg = (
            "\n"
            + f"Bias: {self.model.mean_module.bias.data.squeeze():.4f}\n"
            + f"Weights: {self.model.mean_module.weights.data.squeeze()}\n"
            + f"Outputscale: {self.model.covar_module.outputscale.data.squeeze():.4f}\n"
            + f"Lengthscales: "
            f"{self.model.covar_module.base_kernel.lengthscale.data.squeeze()}"
        )
        if self.learn_noise:
            msg += (
                f"\nLikelihood noise: "
                f"{self.model.likelihood.noise_covar.noise.data.squeeze():.4f}"
            )
        return msg

    def save_to_config_file(self, experiment_file_path):
        config = ConfigParser()
        config["GPExperiment"] = {}
        config["GPExperiment"]["n_restarts"] = str(self.n_restarts)
        config["GPExperiment"]["seed"] = str(self.seed)
        config["GPExperiment"]["learn_noise"] = str(self.learn_noise)

        for i, metric in enumerate(self.metrics):
            config[f"Metric_{i}"] = dump_instance(metric)

        config["Likelihood"] = dump_instance(self.model.likelihood)
        config["Mean"] = dump_instance(self.model.mean_module)
        config["Kernel"] = dump_instance(self.model.covar_module.base_kernel)

        with open(experiment_file_path, "w") as out_f:
            config.write(out_f)


def load_experiment_from_config_file(
    experiment_file_path,
    dataset: Dataset,
) -> GPExperiment:
    config = read_config(experiment_file_path)

    gpexperiment = config["GPExperiment"]
    n_restarts = gpexperiment.getint("n_restarts")
    seed = gpexperiment.get("seed")
    learn_noise = gpexperiment.getboolean("learn_noise")

    if seed == "None":
        seed = None
    else:
        seed = int(seed)
    set_seed(seed)

    metrics = [
        build_instance(**{k: v for k, v in metric.items()})
        for metric in get_repeatable_section(config, "Metric")
    ]

    likelihood = build_instance(**{k: v for k, v in config["Likelihood"].items()})

    mean = build_instance(**{k: v for k, v in config["Mean"].items()})

    kernel = ScaleKernel(build_instance(**{k: v for k, v in config["Kernel"].items()}))
    return GPExperiment(
        dataset,
        likelihood,
        mean,
        kernel,
        n_restarts=n_restarts,
        seed=seed,
        metrics=metrics,
        learn_noise=learn_noise,
    )
