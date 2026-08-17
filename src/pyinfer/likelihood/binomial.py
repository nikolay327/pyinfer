import numpy as np
import scipy.stats as stats

from .base import LikelihoodBase

class Binomial(LikelihoodBase):
    def __init__(self, n: int, p: float, rng_seed=None):
        super(Binomial, self).__init__()
        self.n = n
        self.p = p
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    def __call__(self, k: int):
        if k < 0 or k > self.n:
            raise ValueError("k must be a non-negative integer between 0 and n")

        log_likelihood = stats.binom.logpmf(k, self.n, self.p)
        return log_likelihood

    def sample(self, size=1):
        if self.rng is None:
            sample = np.random.binomial(self.n, self.p, size=size)
        else:
            sample = self.rng.binomial(self.n, self.p, size=size)
        return sample
