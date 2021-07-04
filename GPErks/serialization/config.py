import configparser
from itertools import count

from GPErks.log.logger import get_logger

log = get_logger()


def read_config(config_file_path):
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # preserve case
    config.read(config_file_path)
    log.debug(f"Read configuration file: {config}")
    return config


def get_repeatable_section(config, section_prefix):
    sections = []
    for i in count():
        try:
            sections.append(config[f"{section_prefix}_{i}"])
        except KeyError:
            break
    return sections
