import logging

from GPErks.constants import (
    APPLICATION_NAME,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOGGING_FILE,
)


def get_logger(
    name=APPLICATION_NAME,
    log_format=DEFAULT_LOG_FORMAT,
    stdout_level=logging.INFO,
    file_name=DEFAULT_LOGGING_FILE,
    file_level=logging.INFO,
):
    logging.basicConfig(level=stdout_level, format=log_format)
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    return logger
