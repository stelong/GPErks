import gpytorch
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
