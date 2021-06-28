import json

from typing import Dict, List


class TrainStats:
    def __init__(self, metrics_names):
        self.current_epoch: int = 0
        self.best_epoch: int = 0
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }
        self.val_metrics_score: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metrics_names
        }

    def save_to_file(self, output_file):
        with open(output_file, "w") as out_f:
            json.dump(self.__dict__, out_f)


def load_train_stats_from_file(filename) -> TrainStats:
    train_stats: TrainStats = TrainStats([])
    with open(filename, "r") as in_f:
        for attribute, value in json.load(in_f).items():
            setattr(train_stats, attribute, value)
    return train_stats
