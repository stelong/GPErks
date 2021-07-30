from typing import Tuple

import gpytorch

from GPErks.train.early_stop import EarlyStoppingCriterion


class MyCustomESC(EarlyStoppingCriterion):
    def _reset(self):
        pass

    def _should_stop(self) -> Tuple[bool, float]:
        pass

    def _on_stop(self) -> Tuple[int, gpytorch.models.ExactGP]:
        pass

    def _on_continue(self):
        pass
