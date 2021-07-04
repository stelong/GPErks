import gpytorch
import torch


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
