import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional

import gpytorch
import torch

from GPErks.log.logger import get_logger
from GPErks.train.train_stats import TrainStats

log = get_logger()


class SnapshottingCriterion(metaclass=ABCMeta):
    def __init__(
        self,
        snapshot_dir: str,
        snapshot_file: str,
    ):
        self.model: Optional[gpytorch.models.ExactGP] = None
        self.train_stats: Optional[TrainStats] = None
        self.snapshot_dir: str = snapshot_dir
        self.snapshot_file: str = snapshot_file
        self._reached_epoch: int = -1

    def enable(
        self,
        model: gpytorch.models.ExactGP,
        train_stats: TrainStats,
    ):
        self.model = model
        self.train_stats = train_stats
        self._reached_epoch: int = -1

    def maybe_save(self, restart, epoch):
        if self._should_save():
            self.save(restart, epoch)
        else:
            log.debug("Snapshotting criterion: not calling _save_snapshot()")

    def save(self, restart, epoch):
        log.debug("Snapshotting criterion: calling _save_snapshot()")
        self._save_snapshot(self.get_snapshot_file_path(restart, epoch))
        self._reached_epoch = epoch

    def get_snapshot_file_path(self, restart, epoch):
        restart_dir = Path(self.snapshot_dir.format(restart=restart))
        restart_dir.mkdir(parents=True, exist_ok=True)
        return (
            restart_dir / self.snapshot_file.format(epoch=epoch)
        ).as_posix()

    def keep_snapshots_until(self, restart, epoch):
        for useless_epoch in range(epoch + 1, self._reached_epoch + 1):
            try:
                os.remove(self.get_snapshot_file_path(restart, useless_epoch))
            except FileNotFoundError:
                pass

    @abstractmethod
    def _should_save(self) -> bool:
        pass

    @abstractmethod
    def _save_snapshot(self, snapshot_file_path):
        pass


class EveryNEpochsSnapshottingCriterion(SnapshottingCriterion):
    def __init__(self, snapshot_dir: str, snapshot_file: str, n):
        super().__init__(snapshot_dir, snapshot_file)
        self.n = n

    def _should_save(self) -> bool:
        return self.train_stats.current_epoch % self.n == 0

    def _save_snapshot(self, snapshot_file_path):
        torch.save(self.model.state_dict(), snapshot_file_path)


class EveryEpochSnapshottingCriterion(EveryNEpochsSnapshottingCriterion):
    def __init__(self, snapshot_dir: str, snapshot_file: str):
        super().__init__(snapshot_dir, snapshot_file, 1)


class NeverSaveSnapshottingCriterion(SnapshottingCriterion):
    def _should_save(self) -> bool:
        return False

    def _save_snapshot(self, snapshot_file_path):
        torch.save(self.model.state_dict(), snapshot_file_path)
