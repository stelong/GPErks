import json
from typing import Dict, List

import numpy
from matplotlib import pyplot as plt

from GPErks.constants import HEIGHT, WIDTH


class TrainStats:
    def __init__(self, metrics_names):
        self.current_epoch: int = 0
        self.best_epoch: int = 0
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }
        self.val_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }
        self.early_stopping_criterion_evaluations: List[float] = []

    def save_to_file(self, output_file):
        with open(output_file, "w") as out_f:
            json.dump(self.__dict__, out_f)

    def plot(self, with_early_stopping_criterion: bool = False):
        figsize = (2 * WIDTH, 2 * HEIGHT / 4)
        with_val = len(self.val_loss) > 0
        if with_val:
            fig, axes = plt.subplots(
                1,
                1 + len(self.train_metrics_score),
                figsize=figsize,
            )
        else:
            fig, axis = plt.subplots(1, 1, figsize=figsize)
            axes = [axis]

        loss_len = len(self.train_loss)

        axes[0].plot(
            numpy.arange(1, loss_len + 1),
            self.train_loss,
            zorder=1,
            label="training",
        )
        axes[0].axvline(self.best_epoch, c="r", ls="--", lw=0.8, zorder=2)
        axes[0].set_ylabel("Loss", fontsize=12, zorder=1)
        axes[0].set_xlabel("Epoch", fontsize=12)

        if with_early_stopping_criterion:
            axis = axes[0].twinx()
            axis.tick_params(axis="y", labelcolor="C2")
            axis.plot(
                numpy.arange(1, loss_len + 1),
                self.early_stopping_criterion_evaluations,
                c="C2",
            )
            axis.set_ylabel("Criterion", color="C2", fontsize=12)

        if with_val:
            axes[0].plot(
                numpy.arange(1, loss_len + 1),
                self.val_loss,
                zorder=1,
                label="validation",
            )

            for metric_name, axis in zip(
                self.val_metrics_score, axes.flat[1:]
            ):
                axis.plot(
                    numpy.arange(1, loss_len + 1),
                    self.val_metrics_score[metric_name],
                )
                axis.axvline(self.best_epoch, c="r", ls="--", lw=0.8)
                axis.set_xlabel("Epoch", fontsize=12)
                axis.set_ylabel(metric_name, fontsize=12)

        axes[0].legend(loc="upper center")

        fig.tight_layout()
        # TODO: provide otpion to customize plot behaviour (e.g. dump files)
        # plt.savefig(
        #     self.savepath + f"loss_vs_epochs_restart_{self.restart_idx}.pdf",
        #     bbox_inches="tight",
        #     dpi=1000,
        # )
        plt.show()


def load_train_stats_from_file(filename) -> TrainStats:
    train_stats: TrainStats = TrainStats([])
    with open(filename, "r") as in_f:
        for attribute, value in json.load(in_f).items():
            setattr(train_stats, attribute, value)
    return train_stats
