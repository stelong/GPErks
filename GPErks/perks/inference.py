from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GPErks.constants import DEFAULT_INFERENCE_GRID_DIM, HEIGHT, WIDTH
from GPErks.train.emulator import GPEmulator
from GPErks.utils.array import get_minmax, tensorize
from GPErks.utils.metrics import get_metric_name


class Inference:
    def __init__(self, emulator: GPEmulator):
        self.emulator = emulator
        self.X_test = self.emulator.experiment.dataset.X_test
        self.y_test = self.emulator.experiment.dataset.y_test
        self.metrics = self.emulator.experiment.metrics
        self.y_pred_mean, self.y_pred_std = self.emulator.predict(self.X_test)
        self.scores_dct = {}

    def summary(self):
        metrics_names = list(map(get_metric_name, self.metrics))
        metrics_scores = list(
            m(tensorize(self.y_pred_mean), tensorize(self.y_test))
            .cpu()
            .numpy()
            for m in self.metrics
        )
        self.scores_dct = {
            key: val for key, val in zip(metrics_names, metrics_scores)
        }

        df = pd.DataFrame(
            data=np.around(np.array(metrics_scores), decimals=4).reshape(
                -1, 1
            ),
            index=metrics_names,
            columns=["Score"],
        )
        print(df)

    def plot(self):
        fig, axis = plt.subplots(1, 1, figsize=(2 * WIDTH, 2 * HEIGHT / 3))

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

    def interpolate_2Dgrid(
        self,
        f: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        grid_dim: int = DEFAULT_INFERENCE_GRID_DIM,
    ):
        if self.X_test.shape[1] != 2:
            raise ValueError("Not a 2D input!")

        X_train = self.emulator.experiment.dataset.X_train
        minmax = get_minmax(X_train)

        x1 = np.linspace(minmax[0, 0], minmax[0, 1], grid_dim)
        x2 = np.linspace(minmax[1, 0], minmax[1, 1], grid_dim)
        x1, x2 = np.meshgrid(x1, x2)
        X_grid = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))

        y_pred_mean, y_pred_std = self.emulator.predict(X_grid)
        y_pred_mean = y_pred_mean.reshape(grid_dim, grid_dim)
        y_pred_std = y_pred_std.reshape(grid_dim, grid_dim)

        n_subplots = 1
        t0 = (0,)
        t1 = (1,)
        if f is not None:
            n_subplots += 1
            t0 = (0,) + t0
            t1 = (0,) + t1

            y_grid = f(X_grid)
            y_grid = y_grid.reshape(grid_dim, grid_dim)
            err = np.abs(
                np.ones((grid_dim, grid_dim), dtype=float)
                - y_pred_mean / y_grid
            )

        fig, axes = plt.subplots(
            n_subplots, 2, figsize=(2 * WIDTH, 2 * HEIGHT / (6 / n_subplots))
        )

        PC0 = axes[t0].pcolormesh(
            x1,
            x2,
            y_pred_mean,
            cmap=plt.get_cmap("viridis"),
            vmin=y_pred_mean.min(),
            vmax=y_pred_mean.max(),
            shading="auto",
        )
        cbar0 = fig.colorbar(PC0, ax=axes[t0])
        cbar0.set_label("GPE posterior mean", fontsize=12)

        PC1 = axes[t1].pcolormesh(
            x1, x2, y_pred_std, cmap=plt.get_cmap("viridis"), shading="auto"
        )
        cbar1 = fig.colorbar(PC1, ax=axes[t1])
        cbar1.set_label("GPE posterior std", fontsize=12)

        if n_subplots == 2:
            PC2 = axes[1, 0].pcolormesh(
                x1,
                x2,
                y_grid,
                cmap=plt.get_cmap("viridis"),
                vmin=y_pred_mean.min(),
                vmax=y_pred_mean.max(),
                shading="auto",
            )
            cbar2 = fig.colorbar(PC2, ax=axes[1, 0])
            cbar2.set_label("True value", fontsize=12)

            PC3 = axes[1, 1].pcolormesh(
                x1, x2, err, cmap=plt.get_cmap("magma"), shading="auto"
            )
            cbar3 = fig.colorbar(PC3, ax=axes[1, 1])
            cbar3.set_label("Relative Error", fontsize=12)
            axes[1, 1].scatter(
                X_train[:, 0],
                X_train[:, 1],
                fc="r",
                ec="r",
                label="training data",
            )
            axes[1, 1].legend(loc="upper center")

        fig.tight_layout()
        plt.show()
