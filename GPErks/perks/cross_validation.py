from copy import deepcopy
from itertools import cycle
from typing import List

from sklearn.model_selection import KFold

from GPErks.constants import (
    DEFAULT_CROSS_VALIDATION_MAX_WORKERS,
    DEFAULT_CROSS_VALIDATION_N_SPLITS,
    DEFAULT_TRAIN_MAX_EPOCH,
    DEFAULT_TRAIN_SNAPSHOT_DIR,
    DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
    DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
    DEFAULT_TRAIN_SNAPSHOT_SPLIT_TEMPLATE,
)
from GPErks.gp.data.dataset import Dataset
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.perks.inference import Inference
from GPErks.serialization.path import posix_path
from GPErks.train.early_stop import (
    EarlyStoppingCriterion,
    NoEarlyStoppingCriterion,
)
from GPErks.train.emulator import GPEmulator
from GPErks.train.snapshot import (
    NeverSaveSnapshottingCriterion,
    SnapshottingCriterion,
)
from GPErks.train.trainable import Trainable
from GPErks.utils.concurrency import execute_task_in_parallel


log = get_logger()


class KFoldCrossValidation(Trainable):
    def __init__(
        self,
        experiment: GPExperiment,
        devices: List[str],
        n_splits: int = DEFAULT_CROSS_VALIDATION_N_SPLITS,
        max_workers: int = DEFAULT_CROSS_VALIDATION_MAX_WORKERS,
        *,
        shuffle=False,
        random_state=None,
    ):
        self.experiment: GPExperiment = experiment
        self.devices: List[str] = devices
        self.n_splits: int = n_splits
        self.max_workers: int = max_workers
        self.split_generator = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def train(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion = NoEarlyStoppingCriterion(
            DEFAULT_TRAIN_MAX_EPOCH
        ),
        snapshotting_criterion: SnapshottingCriterion = NeverSaveSnapshottingCriterion(
            posix_path(
                DEFAULT_TRAIN_SNAPSHOT_DIR,
                DEFAULT_TRAIN_SNAPSHOT_SPLIT_TEMPLATE,
                DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
            ),
            DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
        ),
    ):
        X = self.experiment.dataset.X_train
        y = self.experiment.dataset.y_train
        splits = {
            i: (
                optimizer,
                early_stopping_criterion,
                snapshotting_criterion,
                i,
                device,
                X[idx_train],
                y[idx_train],
                X[idx_test],
                y[idx_test],
            )
            for i, (device, (idx_train, idx_test)) in enumerate(
                zip(cycle(self.devices), self.split_generator.split(X))
            )
        }
        for split, (best_model, best_train_stats, inference_scores_dct) in execute_task_in_parallel(
            self._train_split, splits, self.max_workers
        ).items():
            print(split)
            print(inference_scores_dct)
            best_train_stats.plot()

    def _train_split(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion,
        snapshotting_criterion: SnapshottingCriterion,
        i,
        device,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        log.info(f"Running K-fold split {i}...")
        dataset = Dataset(
            X_train,
            y_train,
            X_val=self.experiment.dataset.X_val,
            y_val=self.experiment.dataset.y_val,
            X_test=X_test,
            y_test=y_test,
            x_labels=self.experiment.dataset.x_labels,
            y_label=self.experiment.dataset.y_label,
        )
        experiment = GPExperiment(
            dataset,
            deepcopy(self.experiment.likelihood),
            deepcopy(self.experiment.mean_module),
            deepcopy(self.experiment.covar_module),
            n_restarts=self.experiment.n_restarts,
            seed=self.experiment.seed,
            metrics=self.experiment.metrics,
            learn_noise=self.experiment.learn_noise,
        )

        optimizer = deepcopy(optimizer)
        early_stopping_criterion = deepcopy(early_stopping_criterion)
        snapshotting_criterion = deepcopy(snapshotting_criterion)
        snapshotting_criterion.snapshot_dir = (
            snapshotting_criterion.snapshot_dir.replace(
                "{restart}", "{{restart}}"
            ).format(split=i)
        )

        emulator = GPEmulator(
            experiment,
            device,
        )
        best_model, best_train_stats = emulator.train(
            optimizer,
            early_stopping_criterion,
            snapshotting_criterion,
        )
        best_train_stats.plot()
        log.info(f"Run K-fold split {i}.")
        inference = Inference(emulator)
        inference.summary()
        return best_model, best_train_stats, inference.scores_dct
