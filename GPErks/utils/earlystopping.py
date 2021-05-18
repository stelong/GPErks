from abc import ABCMeta, abstractmethod
from typing import Optional, List

import gpytorch
import numpy
import torch

from GPErks.utils.train_stats import TrainStats
from GPErks.utils.log import get_logger

log = get_logger()


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


class SnapshottingEarlyStoppingCriterion(
    EarlyStoppingCriterion,
    metaclass=ABCMeta
):

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
    # note: uses validation data

    def __init__(self, max_epochs: int, alpha: float, patience: int):
        super().__init__(max_epochs)
        self.alpha: float = alpha
        self.patience: int = patience
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []

    def enable(
            self,
            model: gpytorch.models.ExactGP,
            train_stats: TrainStats,
            save_path,
    ):
        super(GLEarlyStoppingCriterion, self).enable(
            model,
            train_stats,
            save_path
        )

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
        log.debug("GLEpochEarlyStoppingCriterion on_continue().")
