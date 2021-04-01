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
