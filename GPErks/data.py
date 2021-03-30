from GPErks.utils.tensor import tensorize


class ScaledData:
    def __init__(self, X_train, y_train, X_scaler, y_scaler):
        self.scx = X_scaler
        self.scx.fit(X_train)
        self.X_train = tensorize(self.scx.transform(X_train))
        self.scy = y_scaler
        self.scy.fit(y_train)
        self.y_train = tensorize(self.scy.transform(y_train))
        self.input_size = self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1

