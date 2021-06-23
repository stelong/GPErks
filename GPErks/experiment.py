from typing import List, Optional

import gpytorch
import numpy
import torch
import torchmetrics

from GPErks.data import ScaledData
from GPErks.models import ExactGPModel
from GPErks.utils.random import set_seed
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
