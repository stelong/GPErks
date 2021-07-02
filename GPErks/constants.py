from pathlib import Path

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
N = 1024
N_DRAWS = 1000
THRESHOLD = 0.01

# plots constants:
HEIGHT = 9.36111
WIDTH = 5.91667
