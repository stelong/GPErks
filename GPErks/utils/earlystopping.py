from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Optional, List

import gpytorch
import numpy
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
    def __init__(self, max_epochs: int):
        self.max_epochs: int = max_epochs
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

    def evaluate(self) -> Optional[int]:
        if self.train_stats.current_epoch == self.max_epochs or \
                self._should_stop():
            log.debug(
                f"Early stopping: calling on_stop() at epoch "
                f"{self.train_stats.current_epoch}"
            )
            self.is_verified = True
            best_epoch = self._on_stop()
            self._reset()
            return best_epoch
        else:
            log.debug("Early stopping: calling on_continue()")
            self._on_continue()
            return None

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _should_stop(self) -> bool:
        pass

    @abstractmethod
    def _on_stop(self) -> int:
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


class NoEarlyStoppingCriterion(SnapshottingEarlyStoppingCriterion):

    def __init__(self, max_epochs: int):
        super().__init__(max_epochs)

    def _reset(self):
        pass

    def _should_stop(self) -> bool:
        return False

    def _on_stop(self) -> int:
        super()._on_stop()
        return self.train_stats.current_epoch

    def _on_continue(self):
        pass


class GLEarlyStoppingCriterion(SnapshottingEarlyStoppingCriterion):
    # ref: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf

    def __init__(self, max_epochs: int, alpha: float, patience: int):
        super().__init__(max_epochs)
        self.alpha: float = alpha
        self.patience: int = patience
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []

    def _reset(self):
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []

    def _should_stop(self) -> bool:
        if self.train_stats.val_loss[-1] < self.Eva_opt:
            self.Eva_opt = self.train_stats.val_loss[-1]
        self.GL.append(
            100 * numpy.abs(1 - self.train_stats.val_loss[-1] / self.Eva_opt)
        )

        log.info(f"GL[-1]={self.GL[-1]}, alpha={self.alpha}")
        if self.GL[-1] > self.alpha:
            self.counter += 1
            log.info(
                f"Triggered GLEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        else:
            if self.counter > 0:
                log.info(f"Resetting GLEpochEarlyStoppingCriterion countdown.")
            log.info(
                f"The best epoch I will return is: {self.train_stats.current_epoch}"
            )
            self.counter = 0
        return self.counter == self.patience

    def _on_stop(self) -> int:
        super()._on_stop()
        log.info("GLEpochEarlyStoppingCriterion on_stop().")
        return self.train_stats.current_epoch - self.patience

    def _on_continue(self):
        # log.info("GLEpochEarlyStoppingCriterion on_continue().")
        pass
