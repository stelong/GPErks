import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import botorch
import gpytorch
import gpytorch.constraints
import numpy
import torch
import torchmetrics

from GPErks.constants import (
    DEFAULT_GSA_N_DRAWS,
    DEFAULT_TRAIN_MAX_EPOCH,
    DEFAULT_TRAIN_SNAPSHOT_DIR,
    DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
    DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
)
from GPErks.gp.data.scaled_data import ScaledData
from GPErks.gp.experiment import GPExperiment
from GPErks.log.logger import get_logger
from GPErks.serialization.path import posix_path
from GPErks.train.early_stop import (
    EarlyStoppingCriterion,
    NoEarlyStoppingCriterion,
)
from GPErks.train.snapshot import (
    NeverSaveSnapshottingCriterion,
    SnapshottingCriterion,
)
from GPErks.train.train_stats import TrainStats, load_train_stats_from_file
from GPErks.train.trainable import Trainable
from GPErks.utils.array import tensorize
from GPErks.utils.metrics import get_metric_name

log = get_logger()


class GPEmulator(Trainable):
    def __init__(
        self,
        experiment: GPExperiment,
        device: str,
    ):
        self.experiment: GPExperiment = experiment
        self.scaled_data: ScaledData = experiment.scaled_data
        self.device: torch.device = torch.device(device)
        self.learn_noise: bool = experiment.learn_noise

        self.model: gpytorch.models.ExactGP = experiment.model
        if not self.learn_noise:
            self.model.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
            )
            self.model.likelihood.noise = 1e-4
            self.model.likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.init_state = deepcopy(self.model.state_dict())

        self.metrics: List[torchmetrics.Metric] = experiment.metrics

        self.restart_idx = None
        self.criterion = None
        self.best_train_metrics_score: Dict[str, float] = {}
        self.best_val_metrics_score: Dict[str, float] = {}

    def train(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion = NoEarlyStoppingCriterion(
            DEFAULT_TRAIN_MAX_EPOCH
        ),
        snapshotting_criterion: SnapshottingCriterion = NeverSaveSnapshottingCriterion(
            posix_path(
                DEFAULT_TRAIN_SNAPSHOT_DIR,
                DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
            ),
            DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
        ),
    ):
        log.info("Training emulator...")

        X_train = self.scaled_data.X_train.to(self.device)
        y_train = self.scaled_data.y_train.to(self.device)

        X_val = self.scaled_data.X_val
        y_val = self.scaled_data.y_val
        if self.scaled_data.with_val:  # move to device only if available
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        restarts_train_stats: List[TrainStats] = []
        restarts_best_epochs = []

        current_restart = 1
        while current_restart <= self.experiment.n_restarts:
            log.info(f"Running restart {current_restart}...")
            self.restart_idx = current_restart
            restart_train_stats, restart_best_epoch = self._train_once(
                X_train,
                y_train,
                optimizer,
                X_val,
                y_val,
                early_stopping_criterion,
                snapshotting_criterion,
            )
            restarts_train_stats.append(restart_train_stats)
            restarts_best_epochs.append(restart_best_epoch)
            log.info(f"Run restart {current_restart}.")
            current_restart += 1

        log.info("Trained emulator.")

        train_loss_list = [
            train_stats.train_loss[best_epoch - 1]
            for train_stats, best_epoch in zip(
                restarts_train_stats, restarts_best_epochs
            )
        ]
        best_overall_loss_idx = numpy.argmin(train_loss_list)
        if self.scaled_data.with_val:
            val_loss_list = [
                train_stats.val_loss[best_epoch - 1]
                for train_stats, best_epoch in zip(
                    restarts_train_stats, restarts_best_epochs
                )
            ]
            best_overall_loss_idx = numpy.argmin(val_loss_list)

        self.best_train_metrics_score = _get_best_metrics_score(
            restarts_best_epochs,
            best_overall_loss_idx,
            [train_stats.train_metrics_score for train_stats in restarts_train_stats],
        )

        if self.scaled_data.with_val:
            self.best_val_metrics_score = _get_best_metrics_score(
                restarts_best_epochs,
                best_overall_loss_idx,
                [train_stats.val_metrics_score for train_stats in restarts_train_stats],
            )

        best_restart = best_overall_loss_idx + 1
        best_epoch = restarts_best_epochs[best_overall_loss_idx]

        log.info(
            f"Loading best model (restart: {best_restart}, epoch: {best_epoch})..."
        )
        best_model_file = snapshotting_criterion.get_snapshot_file_path(
            best_restart, best_epoch
        )
        best_model = torch.load(
            best_model_file,
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(best_model)
        log.info("Loaded best model.")

        best_model_link = posix_path(
            Path(snapshotting_criterion.snapshot_dir).parent.as_posix(),
            "best_model.pth",
        )
        log.debug(f"Linking best model {best_model_file} to {best_model_link}...")
        try:  # if the symlink exists we have to override it
            os.remove(best_model_link)
        except FileNotFoundError:
            pass  # nothing to do
        os.symlink(best_model_file, best_model_link)
        log.debug("Linked best model.")

        log.info(
            f"Loading best train stats "
            f"(restart: {best_restart}, epoch: {best_epoch})..."
        )
        best_train_stats_file = posix_path(
            Path(best_model_file).parent.as_posix(), "train_stats.json"
        )
        best_train_stats = load_train_stats_from_file(best_train_stats_file)
        log.info("Loaded best train stats.")

        best_train_stats_link = posix_path(
            Path(snapshotting_criterion.snapshot_dir).parent.as_posix(),
            "best_train_stats.json",
        )
        log.debug(
            f"Linking best train stats {best_train_stats_file} to "
            f"{best_train_stats_link}..."
        )
        try:  # if the symlink exists we have to override it
            os.remove(best_train_stats_link)
        except FileNotFoundError:
            pass  # nothing to do
        os.symlink(best_train_stats_file, best_train_stats_link)
        log.debug("Linked best train stats.")

        return best_model, best_train_stats

    def train_auto(self):
        self.model.to(self.device)
        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        log.info("Training emulator...")
        botorch.fit.fit_gpytorch_mll(self.criterion)
        log.info("Trained emulator.")

        log.info("Saving model...")
        snapshot_dir = Path(DEFAULT_TRAIN_SNAPSHOT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file_path = posix_path(DEFAULT_TRAIN_SNAPSHOT_DIR, "best_model.pth")
        torch.save(self.model.state_dict(), snapshot_file_path)
        log.info("Saved model.")

    def _train_once(
        self,
        X_train,
        y_train,
        optimizer,
        X_val,
        y_val,
        early_stopping_criterion: EarlyStoppingCriterion,
        snapshotting_criterion: SnapshottingCriterion,
    ):
        self.model.load_state_dict(self.init_state)

        if self.restart_idx > 0:
            # TODO: make range customizable + restart acts also on mean_module params?
            theta_inf, theta_sup = numpy.log(1e-1), numpy.log(1e1)
            hyperparameters = {
                "covar_module.base_kernel.raw_lengthscale": (theta_sup - theta_inf)
                * torch.rand(self.scaled_data.input_size)
                + theta_inf,
                "covar_module.raw_outputscale": (theta_sup - theta_inf) * torch.rand(1)
                + theta_inf,
            }
            self.model.initialize(**hyperparameters)
            if self.learn_noise:
                self.model.likelihood.initialize(
                    raw_noise=(theta_sup - theta_inf) * torch.rand(1) + theta_inf
                )

        self.model.to(self.device)  # train model on user-defined device e.g., GPU

        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

        train_stats = TrainStats(list(map(get_metric_name, self.metrics)))
        early_stopping_criterion.enable(
            self.model,
            train_stats,
        )
        snapshotting_criterion.enable(
            self.model,
            train_stats,
        )
        best_epoch: Optional[int] = None
        best_model: Optional[gpytorch.models.ExactGP] = None

        max_epochs: int = early_stopping_criterion.max_epochs
        while not early_stopping_criterion.is_verified:
            train_stats.current_epoch += 1
            train_loss = self._train_step(X_train, y_train, optimizer)
            train_stats.train_loss.append(train_loss)
            msg = (
                f"[{train_stats.current_epoch:>{len(str(max_epochs))}}/"
                f"{max_epochs:>{len(str(max_epochs))}}] "
                + f"Training Loss: {train_loss:.4f}"
            )

            snapshotting_criterion.maybe_save(
                self.restart_idx,
                train_stats.current_epoch,
            )

            self.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for metric in self.metrics:
                    metric_name, metric_score = self._evaluate_metric(
                        metric, X_train, y_train
                    )
                    msg += f" - {metric_name}: {metric_score:.4f}"
                    train_stats.train_metrics_score[metric_name].append(
                        float(metric_score)
                    )

                if self.scaled_data.with_val:
                    val_loss = self._val_step(X_val, y_val)
                    train_stats.val_loss.append(val_loss)
                    msg += f" | Validation Loss: {val_loss:.4f}"
                    for metric in self.metrics:
                        metric_name, metric_score = self._evaluate_metric(
                            metric, X_val, y_val
                        )
                        msg += f" - {metric_name}: {metric_score:.4f}"
                        train_stats.val_metrics_score[metric_name].append(
                            float(metric_score)
                        )
            log.info(msg)

            best_epoch, best_model = early_stopping_criterion.evaluate()
            if early_stopping_criterion.is_verified:
                snapshotting_criterion.model = best_model
                snapshotting_criterion.save(self.restart_idx, best_epoch)

        train_stats.best_epoch = best_epoch
        train_stats.save_to_file(
            posix_path(
                snapshotting_criterion.snapshot_dir.format(restart=self.restart_idx),
                "train_stats.json",
            )
        )
        snapshotting_criterion.keep_snapshots_until(self.restart_idx, best_epoch)

        self.model.to(torch.device("cpu"))  # trained model is always returned on CPU

        return train_stats, best_epoch

    def _train_step(self, X_train, y_train, optimizer):
        self.model.train()
        optimizer.zero_grad()
        train_loss = -self.criterion(self.model(X_train), y_train)
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    def _val_step(self, X_val, y_val):
        val_loss = -self.criterion(self.model(X_val), y_val)
        return val_loss.item()

    def _evaluate_metric(self, metric, X, y):
        predictions = self.model.likelihood(self.model(X))
        y_pred_mean = predictions.mean
        metric_name = get_metric_name(metric)
        output = (metric_name,)
        if metric_name == "IndependentStandardError":
            y_pred_std = torch.sqrt(predictions.variance)
            output += (metric(y_pred_mean.cpu(), y_pred_std.cpu(), y.cpu()),)
        else:
            output += (metric(y_pred_mean.cpu(), y.cpu()),)

        return output

    def predict(self, X_new, with_covar=False):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # TODO: check off-diagonal elements during fast-pred
            predictions = self.model.likelihood(self.model(X_new))
            y_mean = predictions.mean.numpy()
            y_std = numpy.sqrt(predictions.variance.numpy())
            if with_covar:
                y_covar = predictions.covariance_matrix.numpy()
                covar_sign = numpy.sign(y_covar)

        y_mean, y_std = self.scaled_data.scy.inverse_transform(y_mean, ystd_=y_std)
        output = (y_mean, y_std)

        if with_covar:
            # trick here to backtransform a full covariance matrix:
            y_covar_as_vec = numpy.sqrt(
                numpy.abs(
                    y_covar.reshape(
                        len(y_std) ** 2,
                    )
                )
            )
            _, y_covar_as_vec = self.scaled_data.scy.inverse_transform(
                y_mean, ystd_=y_covar_as_vec
            )
            y_covar = covar_sign * numpy.power(
                y_covar_as_vec.reshape(len(y_std), len(y_std)), 2
            )
            output += (y_covar,)

        return output

    def sample(self, X_new: numpy.ndarray, n_draws: int = DEFAULT_GSA_N_DRAWS):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(X_new))
            y_std = numpy.sqrt(predictions.variance.numpy())
            y_samples = predictions.sample(sample_shape=torch.Size([n_draws])).numpy()

        for i in range(n_draws):
            y_samples[i] = self.scaled_data.scy.inverse_transform(
                y_samples[i], ystd_=y_std
            )[0]

        return y_samples

    def hyperparameters(self):
        torch.set_printoptions(sci_mode=False)
        msg = (
            "\n"
            + f"Bias: {self.model.mean_module.bias.data.squeeze():.4f}\n"
            + f"Weights: {self.model.mean_module.weights.data.squeeze()}\n"
            + f"Outputscale: {self.model.covar_module.outputscale.data.squeeze():.4f}\n"
            + f"Lengthscales: "
            f"{self.model.covar_module.base_kernel.lengthscale.data.squeeze()}"
        )
        if self.learn_noise:
            msg += (
                f"\nLikelihood noise: "
                f"{self.model.likelihood.noise_covar.noise.data.squeeze():.4f}"
            )
        print(msg)

    def load_state(
        self,
        model_path: str,
    ):
        log.info("Loading model from hyperparameters state dictionary...")
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        log.info("Loaded model.")


def _get_best_metrics_score(
    restarts_best_epochs,
    best_overall_loss_idx,
    metrics_scores: List[Dict[str, List[float]]],
):
    metrics_score_list = defaultdict(list)
    for metrics_score, best_epoch in zip(metrics_scores, restarts_best_epochs):
        for metric_name, metric_values in metrics_score.items():
            metrics_score_list[metric_name].append(metric_values[best_epoch - 1])
    return {
        metric_name: best_values[best_overall_loss_idx]
        for metric_name, best_values in metrics_score_list.items()
    }
