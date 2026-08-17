import numpy as np
import scipy.stats as stats

from .base import LikelihoodBase

class Binomial(LikelihoodBase):
    def __init__(self, n: int, p: float):
        super(Binomial, self).__init__()

        if n < 0:
            raise ValueError("n must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.n = n
        self.p = p

    def __call__(self, k: int):
        if k < 0 or k > self.n:
            raise ValueError("k must be between 0 and n")
        return stats.binom.logpmf(k, self.n, self.p)

    def sample(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.binomial(self.n, self.p, size=size)
