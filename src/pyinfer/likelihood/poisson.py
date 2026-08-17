import numpy as np
import scipy.stats as stats

from .base import LikelihoodBase

class Poisson(LikelihoodBase):
    def __init__(self, lam: float, rng_seed=None):
        super(Poisson, self).__init__()
        self.lam = lam
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    def __call__(self, k: int):
        if k < 0:
            raise ValueError("k must be a non-negative integer")

        log_likelihood = stats.poisson.logpmf(k, self.lam)
        return log_likelihood

    def sample(self, size=1):
        if self.rng is None:
            sample = np.random.poisson(self.lam, size=size)
        else:
            sample = self.rng.poisson(self.lam, size=size)
        return sample
