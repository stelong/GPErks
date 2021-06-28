import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GPErks.utils.metrics import get_metric_name
from GPErks.utils.tensor import tensorize


class Inference:
    def __init__(self, emulator, X_test, y_test, metrics):
        self.emulator = emulator
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = metrics
        self.y_pred_mean, self.y_pred_std = self.emulator.predict(self.X_test)

    def summary(self):
        metrics_names = list(map(get_metric_name, self.metrics))
        metrics_scores = list(
            m(tensorize(self.y_pred_mean), tensorize(self.y_test)).cpu()
            for m in self.metrics
        )

        df = pd.DataFrame(
            data=np.around(np.array(metrics_scores), decimals=4).reshape(
                -1, 1
            ),
            index=metrics_names,
            columns=["Score"],
        )
        print(df)

    def plot(self, is_input_2D=False):
        height = 9.36111
        width = 5.91667
        fig, axis = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 3))

        l = np.argsort(
            self.y_pred_mean
        )  # let's sort predicted values for a better visualisation
        x = np.arange(len(l))

        ci = 2  # ~95% confidence interval

        axis.scatter(
            x,
            self.y_test[l],
            facecolors="none",
            edgecolors="C0",
            label="observed",
        )
        axis.scatter(
            x,
            self.y_pred_mean[l],
            facecolors="C0",
            s=16,
            label="predicted",
        )
        axis.errorbar(
            x,
            self.y_pred_mean[l],
            yerr=ci * self.y_pred_std[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} STD)",
        )

        axis.set_xticks([])
        axis.set_xticklabels([])
        axis.legend(loc="upper left")

        fig.tight_layout()
        plt.show()

    def interpolate_2Dgrid(self, f=None):
        if self.X_test.shape[1] != 2:
            raise ValueError("Not a 2D input!")

        n = 50
        X_train = self.emulator.experiment.scaled_data.X_train
        minmax = get_minmax(X_train)

        x1 = np.linspace(minmax[0, 0], minmax[0, 1], n)
        x2 = np.linspace(minmax[1, 0], minmax[1, 1], n)
        x1, x2 = np.meshgrid(x1, x2)
        X_grid = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))

        y_pred_mean, y_pred_std = self.emulator.predict(X_grid)
        y_pred_mean = y_pred_mean.reshape(n, n)
        y_pred_std = y_pred_std.reshape(n, n)

        n_subplots = 2
        if f is not None:
            n_subplots = 3
            y_grid = f(X_grid)
            y_grid = y_grid.reshape(n, n)
            err = np.abs(np.ones((n, n), dtype=float) - y_pred_mean / y_grid)

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(
            1, n_subplots, figsize=(2 * width, 2 * height / 6)
        )

        PC0 = axes[0].pcolormesh(
            x1, x2, y_pred_mean, cmap=plt.get_cmap("viridis"), shading="auto"
        )
        cbar0 = fig.colorbar(PC0, ax=axes[0])
        cbar0.set_label("GPE posterior mean", fontsize=12)

        PC1 = axes[1].pcolormesh(
            x1, x2, y_pred_std, cmap=plt.get_cmap("viridis"), shading="auto"
        )
        cbar1 = fig.colorbar(PC1, ax=axes[1])
        cbar1.set_label("GPE posterior std", fontsize=12)

        if n_subplots == 3:
            PC2 = axes[2].pcolormesh(
                x1, x2, err, cmap=plt.get_cmap("magma"), shading="auto"
            )
            cbar2 = fig.colorbar(PC2, ax=axes[2])
            cbar2.set_label("Relative Error", fontsize=12)
            axes[2].scatter(X_train[:, 0], X_train[:, 1], fc="r", ec="r")

        fig.tight_layout()
        plt.show()


def get_minmax(X):
    minmax = []
    for x in X.T:
        minmax.append([x.min(), x.max()])
    return np.array(minmax)
