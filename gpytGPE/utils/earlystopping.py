import numpy as np
import torch


def analyze_losstruct(loss):
    counts, edges = np.histogram(loss, bins="sqrt")
    delta = edges[1] - edges[0]

    interval = {}
    for i in range(len(counts)):
        interval[i] = [edges[i], edges[i + 1]]

    l = np.argsort(counts)
    mp_idx = l[-1]
    in_most_populated_interval = interval[mp_idx]

    def is_val(x, in_interval):
        return in_interval[0] <= x and x <= in_interval[1]

    for i, x in enumerate(loss):
        if is_val(x, in_most_populated_interval):
            bellepoque = i
            break

    if mp_idx != 0:
        in_lowest_loss_interval = interval[0]
        c = 0
        for x in loss[bellepoque:]:
            if is_val(x, in_lowest_loss_interval):
                bellepoque += c
                break
            else:
                c += 1

    return bellepoque, 0.5 * delta


class EarlyStopping:  # credits: https://github.com/Bjarten/early-stopping-pytorch
    def __init__(self, patience, delta, savepath):
        self.patience = patience
        self.delta = delta
        self.savepath = savepath

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.savepath + "checkpoint.pth")
        self.val_loss_min = val_loss
