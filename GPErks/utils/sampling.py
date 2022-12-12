from typing import Optional

from scipy.stats import qmc

from GPErks.utils.random import RandomEngine


class Sampler:
    def __init__(
        self,
        design: str = "lhs",
        dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.design = design
        if design == "srs":
            self.engine = RandomEngine(d=dim, seed=seed)
        elif design == "lhs":
            self.engine = qmc.LatinHypercube(d=dim, seed=seed)
        elif design == "sobol":
            self.engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        else:
            raise ValueError(
                "Not a valid sampling design! Choose between 'srs', 'lhs', 'sobol'"
            )

    def sample(self, n_samples, l_bounds=None, u_bounds=None):
        x = self.engine.random(n=n_samples)
        if l_bounds is not None and u_bounds is not None:
            x = qmc.scale(x, l_bounds, u_bounds)
        return x
