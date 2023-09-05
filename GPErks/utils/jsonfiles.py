import json
from pathlib import Path

import numpy

from GPErks.serialization.labels import read_labels_from_file


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(dct, filename):
    with open(filename, "w") as f:
        json.dump(dct, f, cls=NumpyEncoder, indent=4)
    return


def numpy_hook(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            value = numpy.array(value)
            dct[key] = value
    return dct


def load_json(filename):
    dct = {}
    with open(filename, "r") as f:
        dct = json.load(f, object_hook=numpy_hook)
    return dct


def create_json_dataset_from_arrays(data_dir: Path):
    X_train = numpy.loadtxt(data_dir / "X_train.txt", dtype=float)
    Y_train = numpy.loadtxt(data_dir / "Y_train.txt", dtype=float)
    xlabels = read_labels_from_file(data_dir / "xlabels.txt")
    ylabels = read_labels_from_file(data_dir / "ylabels.txt")

    # TODO: use path.glob(*.txt) to detect whether the user has got also val and test
    data_dct = {
        "X_train": X_train.tolist(),
        "Y_train": Y_train.tolist(),
        # "X_val": X_val.tolist(),
        # "Y_val": Y_val.tolist(),
        # "X_test": X_test.tolist(),
        # "Y_test": Y_test.tolist(),
        "x_labels": xlabels,
        "y_labels": ylabels,
        "info": "This is the most beautiful dataset you'll ever see!",
    }
    with open(data_dir / f"{data_dir.name}.json", "w") as f:
        json.dump(data_dct, f, indent=4)
