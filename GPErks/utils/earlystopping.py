from abc import ABCMeta, abstractmethod
from typing import Optional

import gpytorch
import numpy as np
import torch

from GPErks.utils.train_stats import TrainStats
from GPErks.utils.log import get_logger

log = get_logger()


# def analyze_losstruct(loss):
#     counts, edges = np.histogram(loss, bins="sqrt")
#     delta = edges[1] - edges[0]
#
#     interval = {}
#     for i in range(len(counts)):
#         interval[i] = [edges[i], edges[i + 1]]
#
#     l = np.argsort(counts)
#     mp_idx = l[-1]
#     in_most_populated_interval = interval[mp_idx]
#
#     def is_val(x, in_interval):
#         return in_interval[0] <= x and x <= in_interval[1]
#
#     for i, x in enumerate(loss):
#         if is_val(x, in_most_populated_interval):
#             bellepoque = i
#             break
#
#     if mp_idx != 0:
#         in_lowest_loss_interval = interval[0]
#         c = 0
#         for x in loss[bellepoque:]:
#             if is_val(x, in_lowest_loss_interval):
#                 bellepoque += c
#                 break
#             else:
#                 c += 1
#
#     return bellepoque, 0.5 * delta


# class EarlyStopping:  # credits: https://github.com/Bjarten/early-stopping-pytorch
#     def __init__(self, patience, delta, savepath):
#         self.patience = patience
#         self.delta = delta
#         self.savepath = savepath
#
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#
#     def __call__(self, val_loss, model):
#         score = -val_loss
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(
#                 f"EarlyStopping counter: {self.counter} out of {self.patience}"
#             )
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         torch.save(model.state_dict(), self.savepath + "checkpoint.pth")
#         self.val_loss_min = val_loss


class EarlyStoppingCriterion(metaclass=ABCMeta):
    def __init__(self):
        self.model: Optional[gpytorch.models.ExactGP] = None
        self.train_stats: Optional[TrainStats] = None
        self.save_path = None
        self.is_verified = False

    def enable(
            self,
            model: gpytorch.models.ExactGP,
            train_stats: TrainStats,
            save_path,
    ):
        self.model = model
        self.train_stats = train_stats
        self.save_path = save_path
        self.is_verified = False

    def evaluate(self):
        if self._should_stop():
            log.debug("Early stopping: calling on_stop()")
            self.is_verified = True
            self._on_stop()
        else:
            log.debug("Early stopping: calling on_continue()")
            self._on_continue()

    @abstractmethod
    def _should_stop(self) -> bool:
        pass

    @abstractmethod
    def _on_stop(self):
        pass

    @abstractmethod
    def _on_continue(self):
        pass


class SnapshottingEarlyStoppingCriterion(EarlyStoppingCriterion, metaclass=ABCMeta):

    def _on_stop(self):
        log.info("SnapshotEarlyStoppingCriterion on_stop().")
        log.info(f"Saving model to {self.save_path} file...")
        torch.save(self.model.state_dict(), self.save_path)
        log.info(f"Saved model to {self.save_path} file.")


class NoEarlyStoppingCriterion(EarlyStoppingCriterion):

    def _should_stop(self) -> bool:
        return False

    def _on_stop(self):
        pass

    def _on_continue(self):
        pass


class FixedEpochEarlyStoppingCriterion(SnapshottingEarlyStoppingCriterion):

    def __init__(self, epoch: int):
        super().__init__()
        self.epoch: int = epoch

    def _should_stop(self) -> bool:
        return self.train_stats.current_epoch == self.epoch

    def _on_stop(self):
        super()._on_stop()
        log.info("FixedEpochEarlyStoppingCriterion on_stop().")

    def _on_continue(self):
        # log.info("FixedEpochEarlyStoppingCriterion on_continue().")
        pass
