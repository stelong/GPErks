from abc import ABCMeta, abstractmethod

from GPErks.train.early_stop import EarlyStoppingCriterion
from GPErks.train.snapshot import SnapshottingCriterion


class Trainable(metaclass=ABCMeta):
    @abstractmethod
    def train(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion,
        snapshotting_criterion: SnapshottingCriterion,
    ):
        pass
