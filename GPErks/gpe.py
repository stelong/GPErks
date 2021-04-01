from copy import deepcopy

import gpytorch
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy
import torch

from GPErks.data import ScaledData
from GPErks.utils.earlystopping import EarlyStopping, analyze_losstruct
from GPErks.utils.metrics import get_metric_name
from GPErks.utils.tensor import tensorize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LOAD = torch.device("cpu")
FILENAME = "gpe.pth"
LEARN_NOISE = True
LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
N_DRAWS = 1000
N_RESTARTS = 1
PATH = "./"
PATIENCE = 20
SAVE_LOSSES = False
SCALE_DATA = True
WATCH_METRIC = "R2Score"


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
        self.metrics = metrics  # TODO manage multiple metrics

    def train(
        self,
        max_epochs=MAX_EPOCHS,
        n_restarts=N_RESTARTS,
        patience=PATIENCE,
        savepath=PATH,
        save_losses=SAVE_LOSSES,
        watch_metric=WATCH_METRIC,
    ):
        print("\nTraining emulator...")
        self.n_restarts = n_restarts
        self.max_epochs = max_epochs
        self.patience = patience
        self.savepath = savepath
        self.save_losses = save_losses
        self.watch_metric = get_metric_name(self.metrics)

        train_loss_list = []
        model_state_list = []
        if self.scaled_data.with_val:
            val_loss_list = []
            metric_score_list = {get_metric_name(m): [] for m in self.metrics}

        self.idx_best_list = []
        i = 0

        X_train = self.scaled_data.X_train.to(self.device)
        y_train = self.scaled_data.y_train.to(self.device)

        if self.scaled_data.with_val:
            X_val = self.scaled_data.X_val.to(self.device)
            y_val = self.scaled_data.y_val.to(self.device)
        else:  # missing validation data: no need to move anything to device
            X_val = self.scaled_data.X_val
            y_val = self.scaled_data.y_val

        while i < n_restarts + 1:
            self.restart_idx = i
            if self.restart_idx == 0:
                print("\nAnalyzing loss structure...")
                self.print_msg = False
                self.delta = 0
                self.bellepoque = self.max_epochs - 1
            else:
                print(f"\nRestart {self.restart_idx}...")
                self.print_msg = True

            try:
                self.train_once(X_train, y_train, X_val, y_val)
            except RuntimeError as err:
                print(
                    f"Repeating restart {self.restart_idx} because of RuntimeError: {err.args[0]}"
                )
            else:
                i += 1

                if self.restart_idx > 0:
                    self.idx_best_list.append(self.idx_best)
                    train_loss_list.append(self.train_loss_list[self.idx_best])
                    model_state_list.append(self.best_model)
                    if self.scaled_data.with_val:
                        val_loss_list.append(self.val_loss_list[self.idx_best])
                        for metric_name in self.metric_score_list.keys():
                            metric_score_list[metric_name].append(
                                self.metric_score_list[metric_name][
                                    self.idx_best
                                ]
                            )

        if self.scaled_data.with_val:
            idx_min = numpy.argmin(val_loss_list)
            self.best_metric_score = [
                metric_score_list[m][idx_min]
                for m in self.metric_score_list.keys()
            ]
        else:
            idx_min = numpy.argmin(train_loss_list)
        self.best_restart = idx_min + 1
        self.best_epoch = self.idx_best_list[idx_min] + 1

        self.best_model = model_state_list[idx_min]
        self.model.load_state_dict(self.best_model)

        print(
            f"\nDone. The best model resulted from Restart {self.best_restart}, Epoch {self.best_epoch}."
        )
        print("\nThe fitted emulator hyperparameters are:")
        self.print_stats()

    def train_once(self, X_train, y_train, X_val, y_val):
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

        early_stopping = EarlyStopping(
            self.patience, self.delta, self.savepath
        )

        self.train_loss_list = []
        if self.scaled_data.with_val:
            self.val_loss_list = []
            self.metric_score_list = {
                get_metric_name(m): [] for m in self.metrics
            }

        for epoch in range(self.max_epochs):
            train_loss = self.train_step(X_train, y_train)
            if self.scaled_data.with_val:
                val_loss, metric_scores = self.val_step(X_val, y_val)

            msg = (
                f"[{epoch+1:>{len(str(self.max_epochs))}}/{self.max_epochs:>{len(str(self.max_epochs))}}] "
                + f"Training Loss: {train_loss:.4f}"
            )
            if self.scaled_data.with_val:
                msg += f" - Validation Loss: {val_loss:.4f}"
                for metric, metric_score in zip(self.metrics, metric_scores):
                    msg += f" - {get_metric_name(metric)}: {metric_score:.4f}"
            if self.print_msg:
                print(msg)

            self.train_loss_list.append(train_loss)
            if self.scaled_data.with_val:
                self.val_loss_list.append(val_loss)
                for metric_name, metric_score in zip(
                    self.metric_score_list.keys(), metric_scores
                ):
                    self.metric_score_list[metric_name].append(metric_score)

            if epoch >= self.bellepoque:
                if self.scaled_data.with_val:
                    early_stopping(val_loss, self.model)
                else:
                    early_stopping(train_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        self.best_model = torch.load(self.savepath + "checkpoint.pth")
        if self.scaled_data.with_val:
            self.idx_best = numpy.argmin(self.val_loss_list)
        else:
            self.idx_best = numpy.argmin(self.train_loss_list)

        if self.restart_idx == 0:
            if self.scaled_data.with_val:
                self.bellepoque, self.delta = analyze_losstruct(
                    numpy.array(self.val_loss_list)
                )
            else:
                self.bellepoque, self.delta = analyze_losstruct(
                    numpy.array(self.train_loss_list)
                )
            print("\nDone. Now the training starts...")

        if self.save_losses:
            self.plot_loss()

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
            for metric, best_metric_score in zip(
                self.metrics, self.best_metric_score
            ):
                msg += f"\n{get_metric_name(metric)}: {best_metric_score:.4f}"
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

    def plot_loss(self):
        ylabels = ["Training loss"]
        vectors = [self.train_loss_list]
        if self.scaled_data.with_val:
            vectors.append(self.val_loss_list)
            ylabels.append("Validation loss")
            for metric in self.metrics:
                vectors.append(self.metric_score_list[get_metric_name(metric)])
                ylabels.append(get_metric_name(metric))
        n = len(vectors)

        # height = 9.36111
        # width = 5.91667
        # fig = plt.figure(figsize=(2 * width / (4 - n), 2 * height / 3))
        fig = plt.figure()
        gs = grsp.GridSpec(1, n)

        for i, v in enumerate(vectors):
            axis = fig.add_subplot(gs[0, i])
            axis.scatter(numpy.arange(1, len(v) + 1), v)
            axis.axvline(self.idx_best + 1, c="r", ls="--", lw=0.8)
            axis.set_xlabel("Epochs", fontsize=12)
            axis.set_ylabel(ylabels[i], fontsize=12)

        fig.tight_layout()
        plt.savefig(
            self.savepath + f"loss_vs_epochs_restart_{self.restart_idx}.pdf",
            bbox_inches="tight",
            dpi=1000,
        )

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
