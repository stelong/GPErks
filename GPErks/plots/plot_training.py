import numpy

# import matplotlib.gridspec as grsp
from matplotlib import pyplot as plt

# plt.switch_backend('TkAgg')
from GPErks.utils.train_stats import TrainStats


def plot_training_stats(train_stats: TrainStats):
    with_val = len(train_stats.val_loss) > 0
    if with_val:
        fig, axes = plt.subplots(1, 1 + len(train_stats.train_metrics_score))
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
    axes[0].axvline(train_stats.best_epoch, c="r", ls="--", lw=0.8, zorder=2)
    axes[0].set_ylabel("Loss", fontsize=12, zorder=1)
    axes[0].set_xlabel("Epoch", fontsize=12)

    if with_val:
        axes[0].plot(
            numpy.arange(1, loss_len + 1),
            train_stats.val_loss,
            zorder=1,
            label="validation",
        )

        for metric_name, axis in zip(
            train_stats.val_metrics_score, axes.flat[1:]
        ):
            # metric_name = get_metric_name(metric)
            axis.plot(
                numpy.arange(1, loss_len + 1),
                train_stats.val_metrics_score[metric_name],
            )
            axis.axvline(train_stats.best_epoch, c="r", ls="--", lw=0.8)
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
