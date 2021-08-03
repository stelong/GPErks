import multiprocessing
from pathlib import Path

from scipy.stats import norm

from GPErks.serialization.path import posix_path
from GPErks.utils.time import now, pretty_str

APPLICATION_NAME = "GPErks"


# LOGGING
DEFAULT_LOG_FILE = (Path.home() / Path(f"{APPLICATION_NAME}.log")).as_posix()
DEFAULT_LOG_FORMAT = (
    "%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:"
    "L%(lineno)d: %(message)s"
)


# TRAINING
DEFAULT_TRAIN_MAX_EPOCH = 100
DEFAULT_TRAIN_SNAPSHOT_FREQUENCY = DEFAULT_TRAIN_MAX_EPOCH / 20
DEFAULT_TRAIN_SNAPSHOT_DIR = posix_path(".", "snapshot", pretty_str(now()))
DEFAULT_TRAIN_SNAPSHOT_SPLIT_TEMPLATE = "split_{split}"
DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE = "restart_{restart}"
DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE = "epoch_{epoch}.pth"


# INFERENCE
DEFAULT_INFERENCE_GRID_DIM = 50


# CROSS VALIDATION
DEFAULT_CROSS_VALIDATION_N_SPLITS = 5
DEFAULT_CROSS_VALIDATION_MAX_WORKERS = multiprocessing.cpu_count()


# GSA
DEFAULT_GSA_CONF_LEVEL = 0.95
DEFAULT_GSA_Z = norm.ppf(
    0.5 + DEFAULT_GSA_CONF_LEVEL / 2
)  # needed for making SALib return the original and NOT the scaled std
DEFAULT_GSA_N = 1024
DEFAULT_GSA_N_BOOTSTRAP = 100
DEFAULT_GSA_N_DRAWS = 1000
DEFAULT_GSA_SKIP_VALUES = 0
DEFAULT_GSA_THRESHOLD = 0.01


# PLOT
HEIGHT = 9.36111
WIDTH = 5.91667
