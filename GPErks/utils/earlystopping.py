from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Any

import gpytorch
import numpy
import torch

from GPErks.utils.log import get_logger
from GPErks.utils.train_stats import TrainStats

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
    ):
        self.model = model
        self.train_stats = train_stats
        self.is_verified = False

    def evaluate(self) -> Tuple[Optional[int], Optional[gpytorch.models.ExactGP]]:
        if (
            self.train_stats.current_epoch == self.max_epochs
            or self._should_stop()
        ):
            log.debug(
                f"Early stopping: calling on_stop() at epoch "
                f"{self.train_stats.current_epoch}"
            )
            self.is_verified = True
            best_epoch, best_model = self._on_stop()
            self._reset()
            return best_epoch, best_model
        else:
            log.debug("Early stopping: calling on_continue()")
            self._on_continue()
            return None, None

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _should_stop(self) -> bool:
        pass

    @abstractmethod
    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        pass

    @abstractmethod
    def _on_continue(self):
        pass


class NoEarlyStoppingCriterion(EarlyStoppingCriterion):
    def __init__(self, max_epochs: int):
        super().__init__(max_epochs)

    def _reset(self):
        pass

    def _should_stop(self) -> bool:
        return False

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        return self.train_stats.current_epoch, self.model

    def _on_continue(self):
        pass


class PkEarlyStoppingCriterion(EarlyStoppingCriterion):
    # ref: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    # note: uses training data

    def __init__(
        self,
        max_epochs: int,
        alpha: float,
        patience: int,
        strip_length: int,
    ):
        super().__init__(max_epochs)
        self.alpha: float = alpha
        self.patience: int = patience
        self.counter: int = 0
        self.strip_length: int = strip_length
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.Pk: List = []
        self.current_best_state_dict: Dict[Any, Any] = {}

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        super(PkEarlyStoppingCriterion, self).enable(
            model, train_stats,
        )

    def _reset(self):
        self.counter: int = 0
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.Pk: List = []

    def _should_stop(self) -> bool:
        self.strip_counter += 1
        if self.strip_counter % self.strip_length == 0:
            self.computation_enabled = True

        if self.computation_enabled:
            self.Pk.append(
                numpy.abs(
                    numpy.sum(
                        self.train_stats.train_loss[-self.strip_length :]
                    )
                    / (
                        self.strip_length
                        * numpy.min(
                            self.train_stats.train_loss[-self.strip_length :]
                        )
                    )
                    - 1
                )
            )
        else:
            self.Pk.append(0.0)

        log.info(f"Pk[-1]={self.Pk[-1]}, alpha={self.alpha}")
        if self.Pk[-1] > self.alpha:
            self.counter += 1
            log.info(
                f"Triggered PkEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        else:
            if self.counter > 0:
                log.info(f"Resetting PkEpochEarlyStoppingCriterion countdown.")
            log.info(
                f"The best epoch I will return is: {self.train_stats.current_epoch}"
            )
            self.current_best_state_dict = deepcopy(self.model.state_dict())
            self.counter = 0
        return self.counter == self.patience

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.info("PkEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("PkEpochEarlyStoppingCriterion on_continue().")


class GLEarlyStoppingCriterion(EarlyStoppingCriterion):
    # ref: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    # note: uses validation data

    def __init__(self, max_epochs: int, alpha: float, patience: int):
        super().__init__(max_epochs)
        self.alpha: float = alpha
        self.patience: int = patience
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []
        self.current_best_state_dict: Dict[Any, Any] = {}

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        super(GLEarlyStoppingCriterion, self).enable(
            model, train_stats,
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
            self.current_best_state_dict = deepcopy(self.model.state_dict())
        return self.counter == self.patience

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.info("GLEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("GLEpochEarlyStoppingCriterion on_continue().")
