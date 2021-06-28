import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import gpytorch

# import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt

# plt.switch_backend('TkAgg')
import numpy
import torch
import torchmetrics

from GPErks.data import ScaledData
from GPErks.experiment import GPExperiment
from GPErks.snapshotting import SnapshottingCriterion
from GPErks.utils.earlystopping import EarlyStoppingCriterion
from GPErks.utils.log import get_logger
from GPErks.utils.metrics import get_metric_name
from GPErks.utils.tensor import tensorize
from GPErks.utils.train_stats import TrainStats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LOAD = torch.device("cpu")
FILENAME = "gpe.pth"
PATH = "./"
SAVE_LOSSES = False

log = get_logger()


class GPEmulator:
    def __init__(
        self,
        experiment: GPExperiment,
        device=DEVICE,
    ):
        self.experiment: GPExperiment = experiment
        self.scaled_data: ScaledData = experiment.scaled_data
        self.device = device
        self.learn_noise: bool = experiment.learn_noise

        if not self.learn_noise:
            self.model.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
            )
            self.model.likelihood.noise = 1e-4
            self.model.likelihood.noise_covar.raw_noise.requires_grad_(False)

        self.model: gpytorch.models.ExactGP = experiment.model
        self.init_state = deepcopy(self.model.state_dict())

        self.metrics: List[torchmetrics.Metric] = experiment.metrics

    def train(
        self,
        optimizer,
        early_stopping_criterion,
        snapshotting_criterion,
        save_losses=SAVE_LOSSES,
    ):
        print("\nTraining emulator...")
        self.save_losses = save_losses

        self.idx_best_list = []

        X_train = self.scaled_data.X_train.to(self.device)
        y_train = self.scaled_data.y_train.to(self.device)

        X_val = self.scaled_data.X_val
        y_val = self.scaled_data.y_val
        if self.scaled_data.with_val:  # move to device only if available
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        restarts_train_stats: List[TrainStats] = []
        # restarts_best_models = []
        restarts_best_epochs = []

        current_restart = 1
        while current_restart <= self.experiment.n_restarts:
            log.info(f"Running restart {current_restart}...")
            self.restart_idx = current_restart
            (restart_train_stats, restart_best_epoch,) = self.train_once(
                X_train,
                y_train,
                optimizer,
                X_val,
                y_val,
                early_stopping_criterion,
                snapshotting_criterion,
            )
            restarts_train_stats.append(restart_train_stats)
            # restarts_best_models.append(restart_best_model)
            restarts_best_epochs.append(restart_best_epoch)
            log.info(f"Run restart {current_restart}.")
            current_restart += 1

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

        train_metrics_score_list = defaultdict(list)
        for train_stats, best_epoch in zip(
            restarts_train_stats, restarts_best_epochs
        ):
            for (
                metric_name,
                metric_values,
            ) in train_stats.train_metrics_score.items():
                train_metrics_score_list[metric_name].append(
                    metric_values[best_epoch - 1]
                )
        self.best_train_metrics_score = {
            metric_name: best_values[best_overall_loss_idx]
            for metric_name, best_values in train_metrics_score_list.items()
        }

        if self.scaled_data.with_val:
            val_metrics_score_list = defaultdict(list)
            for train_stats, best_epoch in zip(
                restarts_train_stats, restarts_best_epochs
            ):
                for (
                    metric_name,
                    metric_values,
                ) in train_stats.val_metrics_score.items():
                    val_metrics_score_list[metric_name].append(
                        metric_values[best_epoch - 1]
                    )
            self.best_val_metrics_score = {
                metric_name: best_values[best_overall_loss_idx]
                for metric_name, best_values in val_metrics_score_list.items()
            }

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
        log.info(
            f"Loaded best model (restart: {best_restart}, epoch: {best_epoch})."
        )
        best_model_link = (
            Path(snapshotting_criterion.snapshot_dir).parent / "best_model.pth"
        ).as_posix()
        log.debug(
            f"Linking best model {best_model_file} to {best_model_link}..."
        )
        try:  # if the symlink exists we have to override it
            os.remove(best_model_link)
        except FileNotFoundError:
            pass  # nothing to do
        os.symlink(best_model_file, best_model_link)
        log.debug(f"Linked best model {best_model_file} to {best_model_link}.")
        log.info("The fitted emulator hyperparameters are:")
        self.experiment.print_stats()

        if self.scaled_data.with_val:
            msg = ""
            for metric_name, best_value in self.best_val_metrics_score.items():
                msg += f"{metric_name}: {best_value:.4f}\n"
            print(msg)

    def train_once(
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
            theta_inf, theta_sup = numpy.log(1e-1), numpy.log(
                1e1
            )  # TODO: make range customizable
            hyperparameters = {
                "covar_module.base_kernel.raw_lengthscale": (
                    theta_sup - theta_inf
                )
                * torch.rand(self.scaled_data.input_size)
                + theta_inf,
                "covar_module.raw_outputscale": (theta_sup - theta_inf)
                * torch.rand(1)
                + theta_inf,
            }
            self.model.initialize(**hyperparameters)
            if self.learn_noise:
                self.model.likelihood.initialize(
                    raw_noise=(theta_sup - theta_inf) * torch.rand(1)
                    + theta_inf
                )

        self.model.to(self.device)

        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

        # restart_model_checkpoint_file = (
        #     f"{self.savepath}restart{self.restart_idx}_checkpoint.pth"
        # )
        train_stats = TrainStats(list(map(get_metric_name, self.metrics)))
        early_stopping_criterion.enable(
            self.model,
            train_stats,
            # restart_model_checkpoint_file,
        )
        snapshotting_criterion.enable(
            self.model,
            train_stats,
        )

        max_epochs: int = early_stopping_criterion.max_epochs
        while True:
            train_stats.current_epoch += 1
            train_loss = self.train_step(X_train, y_train, optimizer)
            train_stats.train_loss.append(train_loss)
            msg = (
                f"[{train_stats.current_epoch:>{len(str(max_epochs))}}/{max_epochs:>{len(str(max_epochs))}}] "
                + f"Training Loss: {train_loss:.4f}"
            )

            snapshotting_criterion.maybe_save(
                self.restart_idx,
                train_stats.current_epoch,
            )

            self.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                metric_scores = self.evaluate_metrics(X_train, y_train)
                for metric, metric_score in zip(self.metrics, metric_scores):
                    metric_name = get_metric_name(metric)
                    msg += f" - {metric_name}: {metric_score:.4f}"
                    train_stats.train_metrics_score[metric_name].append(
                        metric_score
                    )

                if self.scaled_data.with_val:
                    val_loss = self.val_step(X_val, y_val)
                    train_stats.val_loss.append(val_loss)
                    msg += f" | Validation Loss: {val_loss:.4f}"
                    metric_scores = self.evaluate_metrics(X_val, y_val)
                    for metric, metric_score in zip(
                        self.metrics, metric_scores
                    ):
                        metric_name = get_metric_name(metric)
                        msg += f" - {metric_name}: {metric_score:.4f}"
                        train_stats.val_metrics_score[metric_name].append(
                            metric_score
                        )
            log.info(msg)

            best_epoch: Optional[int] = None
            best_model: Optional[gpytorch.models.ExactGP] = None
            best_epoch, best_model = early_stopping_criterion.evaluate()
            if early_stopping_criterion.is_verified:
                snapshotting_criterion.model = best_model
                snapshotting_criterion.save(self.restart_idx, best_epoch)
                break

        snapshotting_criterion.keep_snapshots_until(
            self.restart_idx, best_epoch
        )

        if self.save_losses:
            self.plot_loss(train_stats, best_epoch)

        return train_stats, best_epoch

    def train_step(self, X_train, y_train, optimizer):
        self.model.train()
        optimizer.zero_grad()
        train_loss = -self.criterion(self.model(X_train), y_train)
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    def val_step(self, X_val, y_val):
        val_loss = -self.criterion(self.model(X_val), y_val)
        return val_loss.item()

    def evaluate_metrics(self, X, y):
        predictions = self.model.likelihood(self.model(X))
        y_pred = predictions.mean
        return [m(y_pred, y).cpu() for m in self.metrics]

    def predict(self, X_new, with_covar=False):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new)).to(
            self.device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(X_new))
            y_mean = predictions.mean.cpu().numpy()
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())
            y_covar = (
                predictions.covariance_matrix.cpu().detach().numpy()
            )  # WHY should detach here since other don't need it?!
            covar_sign = numpy.sign(y_covar)

        y_mean, y_std = self.scaled_data.scy.inverse_transform(
            y_mean, ystd_=y_std
        )

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

        output = (y_mean, y_std)
        if with_covar:
            output += (y_covar,)

        return output

    def sample(self, X_new):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new)).to(
            self.device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(X_new))
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())
            y_samples = (
                predictions.sample(
                    sample_shape=torch.Size([self.experiment.n_draws])
                )
                .cpu()
                .numpy()
            )

        for i in range(self.experiment.n_draws):
            y_samples[i] = self.scaled_data.scy.inverse_transform(
                y_samples[i], ystd_=y_std
            )[0]

        return y_samples

    def plot_loss(self, train_stats: TrainStats, best_epoch: int):
        if self.scaled_data.with_val:
            fig, axes = plt.subplots(1, 1 + len(self.metrics))
        else:
            fig, axis = plt.subplots(1, 1)
            axes = [axis]

        # height = 9.36111
        # width = 5.91667
        # figsize = (2 * width / (4 - n), 2 * height / 3))

        loss_len = len(train_stats.train_loss)

        axes[0].plot(
            numpy.arange(1, loss_len + 1),
            train_stats.train_loss,
            zorder=1,
            label="training",
        )
        axes[0].axvline(best_epoch, c="r", ls="--", lw=0.8, zorder=2)
        axes[0].set_ylabel("Loss", fontsize=12, zorder=1)
        axes[0].set_xlabel("Epoch", fontsize=12)

        if self.scaled_data.with_val:
            axes[0].plot(
                numpy.arange(1, loss_len + 1),
                train_stats.val_loss,
                zorder=1,
                label="validation",
            )

            for metric, axis in zip(self.metrics, axes.flat[1:]):
                metric_name = get_metric_name(metric)
                axis.plot(
                    numpy.arange(1, loss_len + 1),
                    train_stats.val_metrics_score[metric_name],
                )
                axis.axvline(best_epoch, c="r", ls="--", lw=0.8)
                axis.set_xlabel("Epoch", fontsize=12)
                axis.set_ylabel(metric_name, fontsize=12)

        axes[0].legend()

        fig.tight_layout()
        # plt.savefig(
        #     self.savepath + f"loss_vs_epochs_restart_{self.restart_idx}.pdf",
        #     bbox_inches="tight",
        #     dpi=1000,
        # )
        plt.show()

    def save(self, filename=FILENAME):
        print("\nSaving trained emulator...")
        torch.save(self.best_model, self.savepath + filename)
        print("\nDone.")
