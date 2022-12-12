import json

import numpy


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
    with open(filename + ".json", "w") as f:
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
    with open(filename + ".json", "r") as f:
        dct = json.load(f, object_hook=numpy_hook)
    return dct
