import gpytorch
import numpy
import torch


class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, data_mean, batch_shape=torch.Size()):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, input_size, 1)
            ),
        )
        self.register_parameter(
            name="bias",
            parameter=torch.nn.Parameter(
                data_mean * torch.ones(*batch_shape, 1)
            ),
        )

    def forward(self, x):
        res = self.bias + x.matmul(self.weights).squeeze(-1)
        return res


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
    ):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
