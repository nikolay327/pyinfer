import numpy as np
import scipy.stats as stats

from .base import LikelihoodBase

class Poisson(LikelihoodBase):
    def __init__(self, lam: float):
        super(Poisson, self).__init__()

        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lam must be finite and non-negative")

        self.lam = lam

    def __call__(self, k: int):
        if k < 0:
            raise ValueError("k must be non-negative")
        return stats.poisson.logpmf(k, self.lam)

    def sample(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.poisson(self.lam, size=size)
