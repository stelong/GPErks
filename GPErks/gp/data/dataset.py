from matplotlib import pyplot as plt


class Dataset:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val=None,
        y_val=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.with_val = X_val is not None and y_val is not None
        self.X_labels = X_labels
        self.Y_labels = Y_labels
        self.target_label_idx = target_label_idx
        self.target_label = self.Y_labels[self.target_label_idx]

    def plot_train(self):
        self._plot(self.X_train, self.y_train)

    def plot_test(self):
        self._plot(self.X_test, self.y_test)

    def plot_val(self):
        self._plot(self.X_val, self.y_val)

    def _plot(self, Xdata, Ydata, color="C0"):
        """Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.
        Args:
                - Xdata: n*m1 matrix
                - Ydata: n*m2 matrix
                - xlabels: list of m1 strings representing the name of X dataset's features
                - ylabels: list of m2 strings representing the name of Y dataset's features.
        """
        height = 9.36111
        width = 5.91667
        sample_dim = Xdata.shape[0]
        in_dim = Xdata.shape[1]
        out_dim = Ydata.shape[1]
        fig, axes = plt.subplots(
            nrows=out_dim,
            ncols=in_dim,
            sharex="col",
            sharey="row",
            figsize=(2 * width, 2 * height / 3),
        )
        for i, axis in enumerate(axes.flatten()):
            axis.scatter(
                Xdata[:, i % in_dim], Ydata[:, i // in_dim], fc=color, ec=color
            )
            inf = min(Xdata[:, i % in_dim])
            sup = max(Xdata[:, i % in_dim])
            mean = 0.5 * (inf + sup)
            delta = sup - mean
            if i // in_dim == out_dim - 1:
                axis.set_xlabel(self.X_labels[i % in_dim])
                axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
            if i % in_dim == 0:
                axis.set_ylabel(self.Y_labels[i // in_dim])
        plt.suptitle("Sample dimension = {} points".format(sample_dim))
        plt.show()
