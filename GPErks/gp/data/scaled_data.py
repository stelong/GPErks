from torch import Tensor

from GPErks.gp.data.dataset import Dataset
from GPErks.utils.array import tensorize


class ScaledData:
    def __init__(
        self,
        dataset: Dataset,
        X_scaler,
        y_scaler,
    ):
        # TODO: automatic reshape when X_train has size 1 (one point)
        self.scx = X_scaler
        self.scx.fit(dataset.X_train)
        self.X_train: Tensor = tensorize(self.scx.transform(dataset.X_train))
        self.scy = y_scaler
        self.scy.fit(dataset.y_train)
        self.y_train: Tensor = tensorize(self.scy.transform(dataset.y_train))
        self.sample_size = (
            self.X_train.shape[0]
            if len(self.X_train.shape) > 1
            else len(dataset.X_train)
        )
        self.input_size = (
            dataset.input_size
        )  # do we use ScaledData.input_size somewhere?

        self.X_val = None
        self.y_val = None
        self.with_val = dataset.with_val
        if self.with_val:
            self.X_val: Tensor = tensorize(self.scx.transform(dataset.X_val))
            self.y_val: Tensor = tensorize(self.scy.transform(dataset.y_val))
