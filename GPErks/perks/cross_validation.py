from copy import deepcopy
from itertools import cycle
from typing import List, Optional

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
from GPErks.utils.metrics import get_metric_name

log = get_logger()


class KFoldCrossValidation(Trainable):
    def __init__(
        self,
        experiment: GPExperiment,
        devices: List[str],
        n_splits: int = DEFAULT_CROSS_VALIDATION_N_SPLITS,
        max_workers: int = DEFAULT_CROSS_VALIDATION_MAX_WORKERS,
        *,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        leftout_is_val: Optional[bool] = None,
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
        self.leftout_is_val = leftout_is_val

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
        leftout_is_val: bool = False,
    ):
        self.leftout_is_val = leftout_is_val

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
                X[idx_leftout],
                y[idx_leftout],
            )
            for i, (device, (idx_train, idx_leftout)) in enumerate(
                zip(cycle(self.devices), self.split_generator.split(X))
            )
        }  # TODO: dump to file used indices to reproduce splitting outside here

        best_model_dct = {}
        best_train_stats_dct = {}
        inference_scores_dct = {}

        for split, (
            best_model,
            best_train_stats,
            emulator,
        ) in execute_task_in_parallel(
            self._train_split, splits, self.max_workers
        ).items():
            inference = Inference(emulator)
            inference.summary()

            best_model_dct[split] = best_model
            best_train_stats_dct[split] = best_train_stats
            inference_scores_dct[split] = inference.scores_dct

        test_scores = {get_metric_name(m): [] for m in self.experiment.metrics}
        for split, score in inference_scores_dct.items():
            for m in test_scores.keys():
                test_scores[m].append(score[m].item())

        return best_model_dct, best_train_stats_dct, test_scores

    def _train_split(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion,
        snapshotting_criterion: SnapshottingCriterion,
        i,
        device,
        X_train,
        y_train,
        X_leftout,
        y_leftout,
    ):
        log.info(f"Running K-fold split {i}...")

        if self.leftout_is_val:
            dataset = Dataset(
                X_train,
                y_train,
                X_val=X_leftout,
                y_val=y_leftout,
                X_test=X_leftout,
                y_test=y_leftout,
                x_labels=self.experiment.dataset.x_labels,
                y_label=self.experiment.dataset.y_label,
            )
        else:
            dataset = Dataset(
                X_train,
                y_train,
                X_val=self.experiment.dataset.X_val,
                y_val=self.experiment.dataset.y_val,
                X_test=X_leftout,
                y_test=y_leftout,
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

        optimizer = optimizer.__class__(
            experiment.model.parameters(), **optimizer.defaults
        )
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
        log.info(f"Run K-fold split {i}.")

        return best_model, best_train_stats, emulator
