import torch


def R2Score(y, y_pred):
    n_samples = y.size()[0]
    sum_of_sq_errors = torch.sum(torch.pow(y - y_pred, 2)).item()
    y_sq_sum = torch.sum(torch.pow(y, 2)).item()
    y_sum = torch.sum(y).item()
    return 1.0 - sum_of_sq_errors / (y_sq_sum - (y_sum ** 2) / n_samples)


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


def MSE(y, y_pred):
    n_samples = y.size()[0]
    squared_errors = torch.pow(y - y_pred, 2)
    sum_of_sq_errors = torch.sum(squared_errors).item()
    return sum_of_sq_errors / n_samples
