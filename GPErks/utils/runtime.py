import importlib
from ast import literal_eval

from GPErks.utils.log import get_logger

log = get_logger()


def build_instance(*args, **kwargs):
    if "class" not in kwargs:
        raise ValueError(f"Unknown class for object {kwargs}")
    full_class_name = kwargs["class"]
    params = {k: literal_eval(v) for k, v in kwargs.items() if k != "class"}
    log.debug(f"Building {full_class_name} object with params {params}...")
    module_name, class_name = full_class_name.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    class_obj = getattr(module, class_name)
    instance = class_obj(**params)
    log.debug(f"Built {class_name} object with params {params}.")
    return instance
