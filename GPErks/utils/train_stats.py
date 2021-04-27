from typing import List, Dict

import numpy


class TrainStats:
    def __init__(self, metrics_names):
        self.current_epoch: int = 0
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_metrics_score: Dict[str, List[float]] = {
            metric_name: []
            for metric_name in metrics_names
        }
        self.val_metrics_score: Dict[str, List[float]] = {
            metric_name: []
            for metric_name in metrics_names
        }
        print('')

    @property
    def idx_best(self):
        if len(self.val_loss) > 0:
            return numpy.argmin(self.val_loss)
        else:
            return numpy.argmin(self.train_loss)
