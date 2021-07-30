from pathlib import Path

from scipy.stats import norm

from GPErks.serialization.path import posix_path
from GPErks.utils.time import now, pretty_str

APPLICATION_NAME = "GPErks"

# logging
DEFAULT_LOG_FILE = (Path.home() / Path(f"{APPLICATION_NAME}.log")).as_posix()
DEFAULT_LOG_FORMAT = (
    "%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:"
    "L%(lineno)d: %(message)s"
)

# training
DEFAULT_TRAIN_MAX_EPOCH = 100
DEFAULT_TRAIN_SNAPSHOT_FREQUENCY = DEFAULT_TRAIN_MAX_EPOCH / 20
DEFAULT_TRAIN_SNAPSHOT_DIR = posix_path(".", "snapshot", pretty_str(now()))
DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE = "restart_{restart}"
DEFAULT_TRAIN_SNAPSHOT_FILE_TEMPLATE = "epoch_{epoch}.pth"

# gsa constants:
CONF_LEVEL = 0.95
Z = norm.ppf(
    0.5 + CONF_LEVEL / 2
)  # needed for making SALib return the original and NOT the scaled std
N = 1024
N_BOOTSTRAP = 100
N_DRAWS = 1000
SKIP_VALUES = 0
THRESHOLD = 0.01

# plots constants:
HEIGHT = 9.36111
WIDTH = 5.91667
