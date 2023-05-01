import numpy
import torch
from torch import Tensor
from torchmetrics import Metric


def get_metric_name(metric: Metric) -> str:
    return metric.__class__.__name__


class IndependentStandardError:
    def __init__(self, ci: float = 2.0):
        self.ci: float = ci
        self.total: float = 0.0

    def __call__(
        self, y_pred_mean: Tensor, y_pred_std: Tensor, y_true: Tensor
    ) -> Tensor:
        ise = torch.abs(y_true - y_pred_mean) / y_pred_std
        return torch.from_numpy(
            numpy.array(len(torch.where(ise < self.ci)[0]) / len(ise))
        )
