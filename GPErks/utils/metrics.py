import torch
from torchmetrics import Metric


def get_metric_name(metric: Metric) -> str:
    return metric.__class__.__name__


def IndependentStandardError(y_true, y_pred_mean, y_pred_std):
    ise = torch.abs(y_true - y_pred_mean) / y_pred_std
    return 100.0 * len(torch.where(ise < 2.0)[0]) / len(ise)


def MAPE(y, y_pred):
    n_samples = y.size()[0]
    y_c = y.detach().clone()
    l = torch.where(y == 0)[0].tolist()
    if l:
        nl = list(set(range(n_samples)) - set(l))
        correction = torch.min(torch.abs(y[nl]))
        for idx in l:
            y_c[idx] = correction

    abs_rel_errors = torch.abs((y - y_pred)) / torch.abs(y_c)
    sum_of_abs_rel_errors = torch.sum(abs_rel_errors).item()
    return 100.0 * sum_of_abs_rel_errors / n_samples
