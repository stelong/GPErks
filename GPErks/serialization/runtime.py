import importlib
from ast import literal_eval

from GPErks.log.logger import get_logger

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


def dump_instance(instance):
    log.debug(f"Dumping instance {instance}...")
    dump = {}
    module = instance.__class__.__module__
    class_name = instance.__class__.__name__
    full_class_name = f"{module}.{class_name}"
    dump["class"] = full_class_name

    # encode here class-specific logic to dump initialization parameters
    if full_class_name == "gpytorch.means.linear_mean.LinearMean":
        dump["input_size"] = len(instance.weights)
    if full_class_name == "gpytorch.kernels.rbf_kernel.RBFKernel":
        dump["ard_num_dims"] = instance.ard_num_dims

    log.warning(
        f"Dumping instance of class {full_class_name}. "
        f"Please, verify all initialization params are dumped."
    )
    params = [f"{k}={v}" for k, v in dump.items() if k != "class"]
    log.warning(
        f"The object will be built as: {full_class_name}({', '.join(params)})."
    )
    log.debug(f"Dumped instance {instance}.")
    return dump
