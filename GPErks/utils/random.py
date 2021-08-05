import os
import random
from typing import Optional

import numpy
import torch
from scipy.stats import qmc


def set_seed(seed: Optional[int] = None):
    # ref: https://www.kaggle.com/lars123/neural-tangent-kernel-2
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RandomEngine(qmc.QMCEngine):
    def __init__(self, d, seed=None):
        super().__init__(d=d, seed=seed)

    def random(self, n=1):
        self.num_generated += n
        return self.rng.random((n, self.d))

    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self
