import random
from typing import Optional

import numpy
import torch


def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
