import matplotlib.pyplot as plt
import numpy as np

from GPErks.constants import DEFAULT_INFERENCE_GRID_DIM, HEIGHT, WIDTH
from GPErks.train.emulator import GPEmulator
from GPErks.utils.array import tensorize


def inspect_mean_module(
    emulator: GPEmulator,
    grid_dim: int = DEFAULT_INFERENCE_GRID_DIM,
):
    input_size = emulator.experiment.mean_module.input_size

    if input_size > 2:
        raise ValueError(
            "self.input_size is greater than 2! "
            + "You can only inspect the mean module of emulators "
            + "trained on a 1D/2D parameter space."
        )
    else:
        x = emulator.scaled_data.X_train
        y = emulator.scaled_data.y_train

        test_avail = not (emulator.experiment.dataset.X_test is None)
        if test_avail:
            x_test = emulator.scaled_data.scx.transform(
                emulator.experiment.dataset.X_test
            )
            y_test = emulator.scaled_data.scy.transform(
                emulator.experiment.dataset.y_test
            )

        xlabels = emulator.experiment.dataset.x_labels
        ylabel = emulator.experiment.dataset.y_label
        mean_module = emulator.experiment.mean_module

        if input_size == 1:
            x_grid = np.linspace(0, 1, grid_dim**2).reshape(-1, 1)
            y_grid = mean_module.forward(tensorize(x_grid)).cpu().detach().numpy()

            # 1D->1D plot
            fig, axis = plt.subplots(1, 1)
            axis.plot(x_grid, y_grid, c="C0")
            axis.scatter(x, y, fc="C3", label="training data")
            if test_avail:
                axis.scatter(x_test, y_test, fc="C2", label="testing data")
            axis.set_xlabel(xlabels[0])
            axis.set_ylabel(ylabel)

        else:
            x1, x2 = np.meshgrid(
                np.linspace(0, 1, grid_dim), np.linspace(0, 1, grid_dim)
            )
            x_grid = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
            y_grid = mean_module.forward(tensorize(x_grid)).cpu().detach().numpy()
            y_grid = y_grid.reshape(grid_dim, grid_dim)

            # 2D->1D plot
            fig = plt.figure(figsize=(2 * WIDTH, 2 * HEIGHT / 3))
            axis = fig.add_subplot(111, projection="3d")
            axis.plot_surface(
                x1, x2, y_grid, rstride=1, cstride=1, color="b", alpha=0.5
            )
            axis.scatter3D(x[:, 0], x[:, 1], y, c=y, cmap="Reds", label="training data")
            if test_avail:
                axis.scatter3D(
                    x_test[:, 0],
                    x_test[:, 1],
                    y_test,
                    c=y_test,
                    cmap="Greens",
                    label="testing data",
                )
            axis.set_xlabel(xlabels[0])
            axis.set_ylabel(xlabels[1])
            axis.set_zlabel(ylabel)

        axis.legend()
        plt.title(f"LinearMean(degree={mean_module.degree})")
        fig.tight_layout()
        plt.show()
