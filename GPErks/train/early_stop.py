from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import numpy

from GPErks.log.logger import get_logger
from GPErks.train.train_stats import TrainStats

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

    def evaluate(
        self,
    ) -> Tuple[Optional[int], Optional[gpytorch.models.ExactGP]]:
        must_stop = self.train_stats.current_epoch == self.max_epochs

        should_stop = False
        if must_stop:
            # Since self.max_epochs has been reached, there is no need to
            # evaluate the early stopping criterion.
            # In this case, early_stopping_criterion_evaluations will contain
            # (self.max_epochs - 1) items, which is not desirable from a
            # practical point of view (e.g. while plotting data).
            # For convenience, we duplicate the last evaluation.
            self.train_stats.early_stopping_criterion_evaluations.append(
                self.train_stats.early_stopping_criterion_evaluations[-1]
            )
        else:
            should_stop, evaluation = self._should_stop()
            self.train_stats.early_stopping_criterion_evaluations.append(
                evaluation
            )

        if must_stop or should_stop:
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
    def _should_stop(self) -> Tuple[bool, float]:
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

    def _should_stop(self) -> Tuple[bool, float]:
        return False, 0

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
            model,
            train_stats,
        )

    def _reset(self):
        self.counter: int = 0
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.Pk: List = []

    def _should_stop(self) -> Tuple[bool, float]:
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

        log.debug(f"Pk[-1]={self.Pk[-1]}, alpha={self.alpha}")

        if self.computation_enabled and self.Pk[-1] < self.alpha:
            self.counter += 1
            log.debug(
                f"Triggered PkEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        else:
            if self.counter > 0:
                log.debug("Resetting PkEpochEarlyStoppingCriterion countdown.")
            log.debug(
                f"The best epoch I will return is: {self.train_stats.current_epoch}"
            )
            self.current_best_state_dict = deepcopy(self.model.state_dict())
            self.counter = 0
        return self.counter == self.patience, self.Pk[-1]

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.debug("PkEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("PkEpochEarlyStoppingCriterion on_continue().")


class SimpleEarlyStoppingCriterion(EarlyStoppingCriterion):
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    # note: uses validation data

    def __init__(self, max_epochs: int, patience: int):
        super().__init__(max_epochs)
        self.patience: int = patience
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.current_best_state_dict: Dict[Any, Any] = {}

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        super(SimpleEarlyStoppingCriterion, self).enable(
            model,
            train_stats,
        )

    def _reset(self):
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty

    def _should_stop(self) -> Tuple[bool, float]:
        if self.train_stats.val_loss[-1] < self.Eva_opt:
            self.Eva_opt = self.train_stats.val_loss[-1]

            self.current_best_state_dict = deepcopy(self.model.state_dict())
            if self.counter > 0:
                log.debug(
                    "Resetting SimpleEpochEarlyStoppingCriterion countdown."
                )
                self.counter = 0

                log.debug(
                    f"The best epoch I will return is: {self.train_stats.current_epoch}"
                )
        else:
            self.counter += 1
            log.debug(
                f"Triggered SimpleEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        return self.counter == self.patience, self.train_stats.val_loss[-1]

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.debug("SimpleEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("SimpleEpochEarlyStoppingCriterion on_continue().")


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
            model,
            train_stats,
        )

    def _reset(self):
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []

    def _should_stop(self) -> Tuple[bool, float]:
        if self.train_stats.val_loss[-1] < self.Eva_opt:
            self.Eva_opt = self.train_stats.val_loss[-1]
        self.GL.append(
            numpy.abs(100 * (self.train_stats.val_loss[-1] / self.Eva_opt - 1))
        )

        log.debug(f"GL[-1]={self.GL[-1]}, alpha={self.alpha}")
        if self.GL[-1] > self.alpha:
            self.counter += 1
            log.debug(
                f"Triggered GLEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        else:
            if self.counter > 0:
                log.debug("Resetting GLEpochEarlyStoppingCriterion countdown.")
            log.debug(
                f"The best epoch I will return is: {self.train_stats.current_epoch}"
            )
            self.counter = 0
            self.current_best_state_dict = deepcopy(self.model.state_dict())
        return self.counter == self.patience, self.GL[-1]

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.debug("GLEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("GLEpochEarlyStoppingCriterion on_continue().")


class UPEarlyStoppingCriterion(EarlyStoppingCriterion):
    # ref: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    # note: uses validation data

    def __init__(
        self,
        max_epochs: int,
        strip_length: int,
        successive_strips: int,
    ):
        super().__init__(max_epochs)
        self.strip_length: int = strip_length
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.successive_strips: int = successive_strips
        self.sstrips_counter: int = 0
        self.UP: List = []
        self.current_best_state_dict: Dict[Any, Any] = {}

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        super(UPEarlyStoppingCriterion, self).enable(
            model,
            train_stats,
        )

    def _reset(self):
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.sstrips_counter: int = 0
        self.UP: List = []

    def _should_stop(self) -> Tuple[bool, float]:
        self.strip_counter += 1
        if self.strip_counter % self.strip_length == 0:
            self.computation_enabled = True

        if self.computation_enabled:
            self.UP.append(
                numpy.sign(
                    self.train_stats.val_loss[-1]
                    - self.train_stats.val_loss[-self.strip_length]
                )
            )
            if self.UP[-1] == 1:
                self.sstrips_counter += 1
                log.debug(
                    f"Triggered UPEpochEarlyStoppingCriterion {self.sstrips_counter}/{self.successive_strips}"
                )
            else:
                self.sstrips_counter = 0
                self.current_best_state_dict = deepcopy(
                    self.model.state_dict()
                )
        else:
            self.UP.append(0.0)

        log.debug(f"UP[-1]={self.UP[-1]}")

        return self.sstrips_counter == self.successive_strips, self.UP[-1]

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.debug("UPEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return (
            self.train_stats.current_epoch
            - (self.strip_length + self.successive_strips - 1),
            self.model,
        )

    def _on_continue(self):
        log.debug("UPEpochEarlyStoppingCriterion on_continue().")


class PQEarlyStoppingCriterion(EarlyStoppingCriterion):
    # ref: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    # note: uses validation data

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
        self.Eva_opt: float = numpy.infty
        self.GL: List = []
        self.strip_length: int = strip_length
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.Pk: List = []
        self.current_best_state_dict: Dict[Any, Any] = {}
        self.PQ: List = []

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        super(PQEarlyStoppingCriterion, self).enable(
            model,
            train_stats,
        )

    def _reset(self):
        self.counter: int = 0
        self.Eva_opt: float = numpy.infty
        self.GL: List = []
        self.strip_counter: int = 0
        self.computation_enabled: bool = False
        self.Pk: List = []
        self.PQ: List = []

    def _should_stop(self) -> Tuple[bool, float]:
        if self.train_stats.val_loss[-1] < self.Eva_opt:
            self.Eva_opt = self.train_stats.val_loss[-1]
        self.GL.append(
            numpy.abs(100 * (self.train_stats.val_loss[-1] / self.Eva_opt - 1))
        )
        self.strip_counter += 1
        if self.strip_counter % self.strip_length == 0:
            self.computation_enabled = True

        if self.computation_enabled:
            self.Pk.append(
                numpy.abs(
                    1000
                    * (
                        numpy.sum(
                            self.train_stats.train_loss[-self.strip_length :]
                        )
                        / (
                            self.strip_length
                            * numpy.min(
                                self.train_stats.train_loss[
                                    -self.strip_length :
                                ]
                            )
                        )
                        - 1
                    )
                )
            )
            self.PQ.append(self.GL[-1] / self.Pk[-1])
        else:
            self.Pk.append(0.0)
            self.PQ.append(0.0)

        log.debug(f"PQ[-1]={self.PQ[-1]}, alpha={self.alpha}")
        if self.PQ[-1] > self.alpha:
            self.counter += 1
            log.debug(
                f"Triggered PQEpochEarlyStoppingCriterion {self.counter}/{self.patience}"
            )
        else:
            if self.counter > 0:
                log.debug("Resetting PQEpochEarlyStoppingCriterion countdown.")
            log.debug(
                f"The best epoch I will return is: {self.train_stats.current_epoch}"
            )
            self.counter = 0
            self.current_best_state_dict = deepcopy(self.model.state_dict())
        return self.counter == self.patience, self.PQ[-1]

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        super()._on_stop()
        log.debug("PQEpochEarlyStoppingCriterion on_stop().")
        self.model.load_state_dict(self.current_best_state_dict)
        return self.train_stats.current_epoch - self.patience, self.model

    def _on_continue(self):
        log.debug("PQEpochEarlyStoppingCriterion on_continue().")
