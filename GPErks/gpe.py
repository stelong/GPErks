from collections import defaultdict
from copy import deepcopy
from typing import List, Type, Optional

import gpytorch
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import numpy
import torch

from GPErks.data import ScaledData
from GPErks.utils.earlystopping import EarlyStoppingCriterion, \
    NoEarlyStoppingCriterion, GLEarlyStoppingCriterion
from GPErks.utils.log import get_logger
from GPErks.utils.metrics import get_metric_name
from GPErks.utils.tensor import tensorize
from GPErks.utils.train_stats import TrainStats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LOAD = torch.device("cpu")
FILENAME = "gpe.pth"
LEARN_NOISE = True
LEARNING_RATE = 0.1
N_DRAWS = 1000
N_RESTARTS = 4
PATH = "./"
PATIENCE = 20
SAVE_LOSSES = False
SCALE_DATA = True
WATCH_METRIC = "R2Score"


log = get_logger()


class GPEmul:
    def __init__(
        self,
        scaled_data: ScaledData,
        model,
        optimizer,
        metrics,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        learn_noise=LEARN_NOISE,
    ):
        self.scaled_data = scaled_data
        self.device = device
        self.learn_noise = learn_noise

        if not self.learn_noise:
            self.model.likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(1e-6)
            )
            self.model.likelihood.noise = 1e-4
            self.model.likelihood.noise_covar.raw_noise.requires_grad_(False)

        self.model = model
        self.init_state = deepcopy(self.model.state_dict())

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics

    def train(
        self,
        early_stopping_criterion,
        n_restarts=N_RESTARTS,
        patience=PATIENCE,
        savepath=PATH,
        save_losses=SAVE_LOSSES,
    ):
        print("\nTraining emulator...")
        self.n_restarts = n_restarts
        self.patience = patience
        self.savepath = savepath
        self.save_losses = save_losses

        # TODO: clean up (since aggregate values are computed later over restarts' TrainStats)
        train_loss_list = []
        model_state_list = []
        if self.scaled_data.with_val:
            val_loss_list = []
            metric_score_list = {get_metric_name(m): [] for m in self.metrics}

        self.idx_best_list = []

        X_train = self.scaled_data.X_train.to(self.device)
        y_train = self.scaled_data.y_train.to(self.device)

        if self.scaled_data.with_val:
            X_val = self.scaled_data.X_val.to(self.device)
            y_val = self.scaled_data.y_val.to(self.device)
        else:  # missing validation data: no need to move anything to device
            X_val = self.scaled_data.X_val
            y_val = self.scaled_data.y_val

        restarts_train_stats: List[TrainStats] = []
        restarts_best_models = []
        restarts_best_epochs = []

        current_restart = 1
        while current_restart <= n_restarts:
            log.info(f"Running restart {current_restart}...")
            self.restart_idx = current_restart

            # if self.restart_idx == 0:
            #     print("\nAnalyzing loss structure...")
            #     self.print_msg = False
            #     self.delta = 0
            #     self.bellepoque = self.max_epochs - 1
            # else:
            #     print(f"\nRestart {self.restart_idx}...")
            # self.print_msg = True

            # try:
            restart_train_stats, restart_best_model, restart_best_epoch = self.train_once(
                X_train, y_train, X_val, y_val, early_stopping_criterion
            )
            # if self.restart_idx > 0:
            restarts_train_stats.append(restart_train_stats)
            restarts_best_models.append(restart_best_model)
            restarts_best_epochs.append(restart_best_epoch)
            # except RuntimeError as err:
            #     print(
            #         f"Repeating restart {self.restart_idx} because of RuntimeError: {err.args[0]}"
            #     )
            # else:
            log.info(f"Run restart {current_restart}.")
            current_restart += 1

        train_loss_list = [
            train_stats.train_loss[best_epoch]
            for train_stats, best_epoch in zip(restarts_train_stats, restarts_best_epochs) 
        ]
        best_overall_loss_idx = numpy.argmin(train_loss_list)
        if self.scaled_data.with_val:
            val_loss_list = [
                train_stats.val_loss[best_epoch]
                for train_stats, best_epoch in zip(restarts_train_stats, restarts_best_epochs)
            ]
            best_overall_loss_idx = numpy.argmin(val_loss_list)

        # TODO: try if they work AFTER having computed metric values for training data in train_once
        # train_metrics_score_list = defaultdict(list)
        # for rts in restarts_train_stats:
        #     for metric_name, metric_values in rts.train_metrics_score.items():
        #         train_metrics_score_list[metric_name].append(
        #             metric_values[rts.idx_best]
        #         )
        # self.best_train_metrics_score = {
        #     metric_name: best_values[idx_best]
        #     for metric_name, best_values in train_metrics_score_list.items()
        # }

        val_metrics_score_list = None
        if self.scaled_data.with_val:
            val_metrics_score_list = defaultdict(list)
            for train_stats, best_epoch in zip(restarts_train_stats, restarts_best_epochs):
                for metric_name, metric_values in train_stats.val_metrics_score.items():
                    val_metrics_score_list[metric_name].append(
                        metric_values[best_epoch]
                    )
            self.best_val_metrics_score = {
                metric_name: best_values[best_overall_loss_idx]
                for metric_name, best_values in val_metrics_score_list.items()
            }

        # idx_best_list = [rts.idx_best for rts in restarts_train_stats]

        self.best_restart = best_overall_loss_idx + 1
        self.best_epoch = restarts_best_epochs[best_overall_loss_idx] + 1  # TODO: check +1 / old code:  # idx_best_list[best_overall_loss_idx] + 1

        self.best_model = restarts_best_models[best_overall_loss_idx]  # TODO: improve
        self.model.load_state_dict(self.best_model)

        print(
            f"\nDone. The best model resulted from Restart {self.best_restart}, Epoch {self.best_epoch}."
        )
        print("\nThe fitted emulator hyperparameters are:")
        self.print_stats()

    def train_once(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_criterion: EarlyStoppingCriterion,
    ):
        self.model.load_state_dict(self.init_state)

        if self.restart_idx > 0:
            theta_inf, theta_sup = numpy.log(1e-1), numpy.log(1e1)
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

        # early_stopping = EarlyStopping(
        #     self.patience, self.delta, self.savepath
        # )

        restart_model_checkpoint_file = f"{self.savepath}restart{self.restart_idx}_checkpoint.pth"
        train_stats = TrainStats(list(map(get_metric_name, self.metrics)))
        if early_stopping_criterion:
            early_stopping_criterion.enable(
                self.model,
                train_stats,
                restart_model_checkpoint_file,
            )

        max_epochs: int = early_stopping_criterion.max_epochs
        while True:
            train_stats.current_epoch += 1
            train_loss = self.train_step(X_train, y_train)
            train_stats.train_loss.append(train_loss)
            # TODO: compute train_metrics_score
            msg = (
                    f"[{train_stats.current_epoch:>{len(str(max_epochs))}}/{max_epochs:>{len(str(max_epochs))}}] "
                    + f"Training Loss: {train_loss:.4f}"
            )

            if self.scaled_data.with_val:
                val_loss, metric_scores = self.val_step(X_val, y_val)
                train_stats.val_loss.append(val_loss)
                msg += f" - Validation Loss: {val_loss:.4f}"
                for metric, metric_score in zip(self.metrics, metric_scores):
                    metric_name = get_metric_name(metric)
                    msg += f" - {metric_name}: {metric_score:.4f}"
                    train_stats.val_metrics_score[metric_name].append(metric_score)
            # if self.print_msg:
            log.info(msg)

            best_epoch: Optional[int] = early_stopping_criterion.evaluate()
            if early_stopping_criterion.is_verified:
                break

            # if epoch >= self.bellepoque:
            #     if self.scaled_data.with_val:
            #         early_stopping(val_loss, self.model)
            #     else:
            #         early_stopping(train_loss, self.model)
            # if early_stopping.early_stop:
            #     print("Early stopping!")
            #     break

        best_model = torch.load(restart_model_checkpoint_file)
        # if self.restart_idx == 0:
        #     if self.scaled_data.with_val:
        #         self.bellepoque, self.delta = 0, 0.0
        #     else:
        #         self.bellepoque, self.delta = analyze_losstruct(
        #             numpy.array(train_stats.train_loss)
        #         )
        #     print("\nDone. Now the training starts...")

        if self.save_losses:
            self.plot_loss(train_stats, best_epoch)

        return train_stats, best_model, best_epoch

    def train_step(self, X_train, y_train):
        self.model.train()
        self.optimizer.zero_grad()
        train_loss = -self.criterion(self.model(X_train), y_train)
        train_loss.backward()
        self.optimizer.step()

        return train_loss.item()

    def val_step(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_loss = -self.criterion(self.model(X_val), y_val)
            predictions = self.model.likelihood(self.model(X_val))
            y_pred = predictions.mean
            metric_scores = [m(y_pred, y_val).cpu() for m in self.metrics]

        return val_loss.item(), metric_scores

    def print_stats(self):
        torch.set_printoptions(sci_mode=False)
        msg = (
            "\n"
            + f"Bias: {self.model.mean_module.bias.data.squeeze():.4f}\n"
            + f"Weights: {self.model.mean_module.weights.data.squeeze()}\n"
            + f"Outputscale: {self.model.covar_module.outputscale.data.squeeze():.4f}\n"
            + f"Lengthscales: {self.model.covar_module.base_kernel.lengthscale.data.squeeze()}"
        )
        if self.learn_noise:
            msg += f"\nLikelihood noise: {self.model.likelihood.noise_covar.noise.data.squeeze():.4f}"

        if self.scaled_data.with_val:
            for metric_name, best_value in self.best_val_metrics_score.items():
                msg += f"\n{metric_name}: {best_value:.4f}"
        print(msg)

    def predict(self, X_new):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new)).to(
            self.device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(X_new))
            y_mean = predictions.mean.cpu().numpy()
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())

        y_mean, y_std = self.scaled_data.scy.inverse_transform(
            y_mean, ystd_=y_std
        )

        return y_mean, y_std

    def sample(self, X_new, n_draws=N_DRAWS):
        self.model.eval()
        self.model.likelihood.eval()

        X_new = tensorize(self.scaled_data.scx.transform(X_new)).to(
            self.device
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(X_new))
            y_std = numpy.sqrt(predictions.variance.cpu().numpy())
            y_samples = (
                predictions.sample(sample_shape=torch.Size([n_draws]))
                .cpu()
                .numpy()
            )

        for i in range(n_draws):
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

        axes[0].plot(numpy.arange(1, loss_len + 1), train_stats.train_loss, zorder=1, label="training")
        axes[0].axvline(best_epoch, c="r", ls="--", lw=0.8, zorder=2)
        axes[0].set_ylabel("Loss", fontsize=12, zorder=1)
        axes[0].set_xlabel("Epoch", fontsize=12)

        if self.scaled_data.with_val:
            axes[0].plot(numpy.arange(1, loss_len + 1), train_stats.val_loss, zorder=1, label="validation")

            for metric, axis in zip(self.metrics, axes.flat[1:]):
                metric_name = get_metric_name(metric)
                axis.plot(numpy.arange(1, loss_len + 1), train_stats.val_metrics_score[metric_name])
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

    @classmethod
    def load(
        cls,
        data_scaler,
        model,
        optimizer,
        metrics,
        loadpath=PATH,
        filename=FILENAME,
        device=DEVICE_LOAD,
    ):
        print("\nLoading emulator...")
        emul = cls(data_scaler, model, optimizer, metrics, device=device)
        emul.model.load_state_dict(
            torch.load(loadpath + filename, map_location=device)
        )
        emul.model.to(device)
        emul.learn_noise = not numpy.isclose(
            emul.model.likelihood.noise_covar.raw_noise.item(),
            numpy.log(1e-4),
            rtol=0.0,
            atol=1e-1,
        )
        emul.scaled_data.with_val = False

        print("\nDone. The emulator hyperparameters are:")
        emul.print_stats()
        return emul
