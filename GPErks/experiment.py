import random
from typing import Optional, List

import numpy
import gpytorch
import torch
import torchmetrics

from GPErks.data import ScaledData
from GPErks.models.models import ExactGPModel
from GPErks.utils.scaler import UnitCubeScaler, StandardScaler


class GPExperiment:
    def __init__(
            self,
            X_train: numpy.ndarray,
            y_train: numpy.ndarray,
            likelihood: gpytorch.likelihoods.Likelihood,
            mean_module: gpytorch.means.Mean,
            covar_module: gpytorch.kernels.Kernel,
            *,
            metrics: Optional[List[torchmetrics.Metric]] = None,
            X_val: numpy.ndarray = None,
            y_val: numpy.ndarray = None,
            seed: Optional[int] = None,
    ):
        self.scaled_data: ScaledData = ScaledData(
            X_train,
            y_train,
            UnitCubeScaler(),
            StandardScaler(),
            X_val,
            y_val,
        )
        if metrics:
            self.metrics: List[torchmetrics.Metric] = metrics
        else:
            self.metrics: List[torchmetrics.Metric] = []
        if seed:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        self.model: ExactGPModel = ExactGPModel(
            self.scaled_data.X_train,
            self.scaled_data.y_train,
            likelihood,
            mean_module,
            covar_module,
        )
