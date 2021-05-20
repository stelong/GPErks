from typing import Dict, List


class TrainStats:
    def __init__(self, metrics_names):
        self.current_epoch: int = 0
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }
        self.val_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }
