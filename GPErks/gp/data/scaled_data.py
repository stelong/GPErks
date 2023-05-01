from torch import Tensor

from GPErks.gp.data.data_scaler import InputDataScaler, OutputDataScaler
from GPErks.gp.data.dataset import Dataset
from GPErks.utils.array import tensorize


class ScaledData:
    """ScaledData class contains..."""

    def __init__(
        self,
        dataset: Dataset,
        scx: InputDataScaler,
        scy: OutputDataScaler,
    ):
        self.scx = scx
        self.scx.fit(dataset.X_train)
        self.X_train: Tensor = tensorize(self.scx.transform(dataset.X_train))
        self.scy = scy
        self.scy.fit(dataset.y_train)
        self.y_train: Tensor = tensorize(self.scy.transform(dataset.y_train))
        self.sample_size = dataset.sample_size
        self.input_size = dataset.input_size
        self.with_val = dataset.with_val
        self.X_val = dataset.X_val
        self.y_val = dataset.y_val
        if self.with_val:
            self.X_val: Tensor = tensorize(self.scx.transform(self.X_val))
            self.y_val: Tensor = tensorize(self.scy.transform(self.y_val))
