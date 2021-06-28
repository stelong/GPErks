from configparser import ConfigParser
from typing import List, Optional

import gpytorch
import numpy
import torch
import torchmetrics
from gpytorch.kernels import ScaleKernel

from GPErks.data import ScaledData
from GPErks.models import ExactGPModel
from GPErks.utils.config import get_repeatable_section, read_config
from GPErks.utils.random import set_seed
from GPErks.utils.runtime import build_instance, dump_instance
from GPErks.utils.scaler import StandardScaler, UnitCubeScaler


class GPExperiment:
    def __init__(
        self,
        X_train: numpy.ndarray,
        y_train: numpy.ndarray,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        n_restarts: int,
        *,
        seed: Optional[int] = None,
        metrics: Optional[List[torchmetrics.Metric]] = None,
        X_val: numpy.ndarray = None,
        y_val: numpy.ndarray = None,
        n_draws: int = 1000,
        learn_noise: bool = True,
    ):
        set_seed(seed)  # set immediately, for reproducible initialization
        self.seed: Optional[int] = seed

        # scale data by default
        self.scaled_data: ScaledData = ScaledData(
            X_train,
            y_train,
            UnitCubeScaler(),
            StandardScaler(),
            X_val,
            y_val,
        )
        self.n_restarts: int = n_restarts

        if metrics:
            self.metrics: List[torchmetrics.Metric] = metrics
        else:
            self.metrics: List[torchmetrics.Metric] = []

        self.n_draws: int = n_draws
        self.learn_noise: bool = learn_noise

        self.model: ExactGPModel = ExactGPModel(
            self.scaled_data.X_train,
            self.scaled_data.y_train,
            likelihood,
            mean_module,
            covar_module,
        )

    def load_model(
        self,
        model_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.print_stats()

    def print_stats(self):
        torch.set_printoptions(sci_mode=False)
        msg = (
            "\n"
            + f"Bias: {self.model.mean_module.bias.data.squeeze():.4f}\n"
            + f"Weights: {self.model.mean_module.weights.data.squeeze()}\n"
            + f"Outputscale: {self.model.covar_module.outputscale.data.squeeze():.4f}\n"
            + f"Lengthscales: {self.model.covar_module.base_kernel.lengthscale.data.squeeze()}"
        )
        if self.learn_noise:
            msg += f"\nLikelihood noise: {self.model.likelihood.noise_covar.noise.data.squeeze():.4f}"
        print(msg)

    def save_to_config_file(self, experiment_file_path):
        config = ConfigParser()
        config["GPExperiment"] = {}
        config["GPExperiment"]["n_restarts"] = str(self.n_restarts)
        config["GPExperiment"]["seed"] = str(self.seed)
        config["GPExperiment"]["n_draws"] = str(self.n_draws)
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
    X_train: numpy.ndarray,
    y_train: numpy.ndarray,
    X_val: numpy.ndarray = None,
    y_val: numpy.ndarray = None,
) -> GPExperiment:
    config = read_config(experiment_file_path)

    gpexperiment = config["GPExperiment"]
    n_restarts = gpexperiment.getint("n_restarts")
    seed = gpexperiment.getint("seed")
    n_draws = gpexperiment.getint("n_draws")
    learn_noise = gpexperiment.getboolean("learn_noise")

    set_seed(seed)

    metrics = [
        build_instance(**{k: v for k, v in metric.items()})
        for metric in get_repeatable_section(config, "Metric")
    ]

    likelihood = build_instance(
        **{k: v for k, v in config["Likelihood"].items()}
    )

    mean = build_instance(**{k: v for k, v in config["Mean"].items()})

    kernel = ScaleKernel(
        build_instance(**{k: v for k, v in config["Kernel"].items()})
    )
    return GPExperiment(
        X_train,
        y_train,
        likelihood,
        mean,
        kernel,
        n_restarts,
        seed=seed,
        metrics=metrics,
        X_val=X_val,
        y_val=y_val,
        n_draws=n_draws,
        learn_noise=learn_noise,
    )