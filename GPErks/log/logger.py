import logging

from GPErks.constants import (
    APPLICATION_NAME,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_FORMAT,
)


def get_logger(
    name=APPLICATION_NAME,
    log_format=DEFAULT_LOG_FORMAT,
    stdout_level=logging.INFO,
    file_name=DEFAULT_LOG_FILE,
    file_level=logging.DEBUG,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(log_format)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

    return logger
