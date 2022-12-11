import matplotlib.pyplot as plt
import numpy as np

from GPErks.constants import DEFAULT_INFERENCE_GRID_DIM, WIDTH, HEIGHT
from GPErks.train.emulator import GPEmulator
from GPErks.utils.array import get_minmax, tensorize


def inspect_mean_module(
		emulator: GPEmulator,
		grid_dim: int = DEFAULT_INFERENCE_GRID_DIM,
	):
	if emulator.experiment.mean_module.input_size != 2:
		raise ValueError(
			"self.input_size does not equal 2! "
			+ "You can only inspect the mean module of emulators trained on a 2D parameter space."
		)

	X = emulator.scaled_data.X_train
	y = emulator.scaled_data.y_train
	xlabels = emulator.experiment.dataset.x_labels
	ylabel = emulator.experiment.dataset.y_label
	minmax = get_minmax(X)

	x1 = np.linspace(minmax[0, 0], minmax[0, 1], grid_dim)
	x2 = np.linspace(minmax[1, 0], minmax[1, 1], grid_dim)
	x1, x2 = np.meshgrid(x1, x2)
	X_grid = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
	y_grid = (
		emulator.experiment.mean_module.forward(tensorize(X_grid))
		.cpu()
		.detach()
		.numpy()
	)
	y_grid = y_grid.reshape(grid_dim, grid_dim)

	fig = plt.figure(figsize=(2 * WIDTH, 2 * HEIGHT / 3))
	axis = fig.add_subplot(111, projection="3d")

	axis.scatter3D(X[:, 0], X[:, 1], y, c=y, cmap="Reds", label="training data")
	if emulator.experiment.dataset.X_test is not None:
		X_test = emulator.scaled_data.scx.transform(emulator.experiment.dataset.X_test)
		y_test = emulator.scaled_data.scy.transform(emulator.experiment.dataset.y_test)
		axis.scatter3D(
			X_test[:, 0],
			X_test[:, 1],
			y_test,
			c=y_test,
			cmap="Greens",
			label="testing data",
		)

	axis.plot_surface(x1, x2, y_grid, rstride=1, cstride=1, color="b", alpha=0.5)

	axis.legend()
	axis.set_xlabel(xlabels[0])
	axis.set_ylabel(xlabels[1])
	axis.set_zlabel(ylabel)

	plt.title(
		f"LinearMean(degree={emulator.experiment.mean_module.degree})"
	)
	fig.tight_layout()
	plt.show()
