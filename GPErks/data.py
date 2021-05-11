from torch import Tensor

from GPErks.utils.tensor import tensorize


class ScaledData:
    def __init__(
        self, X_train, y_train, X_scaler, y_scaler, X_val=None, y_val=None
    ):
        # TODO: automatic reshape when X_train has size 1 (one point)
        self.scx = X_scaler
        self.scx.fit(X_train)
        self.X_train: Tensor = tensorize(self.scx.transform(X_train))
        self.scy = y_scaler
        self.scy.fit(y_train)
        self.y_train: Tensor = tensorize(self.scy.transform(y_train))
        self.input_size = (
            self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1
        )
        self.X_val = None
        self.y_val = None
        self.with_val = X_val is not None and y_val is not None
        if self.with_val:
            self.X_val: Tensor = tensorize(self.scx.transform(X_val))
            self.y_val: Tensor = tensorize(self.scy.transform(y_val))
