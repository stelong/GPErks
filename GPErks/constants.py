from pathlib import Path

from scipy.stats import norm

APPLICATION_NAME = "GPErks"

# logging
DEFAULT_LOGGING_FILE = (
    Path.home() / Path(f"{APPLICATION_NAME}.log")
).as_posix()
DEFAULT_LOG_FORMAT = (
    "%(levelname)s:%(asctime)s:%(module)s:%(funcName)s:"
    "L%(lineno)d: %(message)s"
)

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
