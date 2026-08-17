import numpy as np

from .base import LikelihoodBase, Sequential
from .poisson import Poisson


class _Fail(LikelihoodBase):
    def __init__(
        self,
        lam_S: float,
        lam_B: float,
        eps_S: float,
        eps_B: float
    ):
        super(_Fail, self).__init__()
        self.update(lam_S, lam_B, eps_S, eps_B)

    def update(self, lam_S: float, lam_B: float, eps_S: float, eps_B: float):
        self.likelihood = Poisson((1 - eps_S) * lam_S + (1 - eps_B) * lam_B)

    def __call__(self, k: int):
        return self.likelihood(k)

    def sample(self, size=1):
        return self.likelihood.sample(size=size)


class _Pass(LikelihoodBase):
    def __init__(
        self,
        lam_S: float,
        lam_B: float,
        eps_S: float,
        eps_B: float
    ):
        super(_Pass, self).__init__()
        self.update(lam_S, lam_B, eps_S, eps_B)

    def update(self, lam_S: float, lam_B: float, eps_S: float, eps_B: float):
        self.likelihood = Poisson(eps_S * lam_S + eps_B * lam_B)

    def __call__(self, k: int):
        return self.likelihood(k)

    def sample(self, size=1):
        return self.likelihood.sample(size=size)

class OneBin(LikelihoodBase):
    def __init__(
        self,
        lam_S: float,
        lam_B: float,
        eps_S: float,
        eps_B: float
    ):
        super(OneBin, self).__init__()
        self.update(lam_S, lam_B, eps_S, eps_B)

    def update(self, lam_S: float, lam_B: float, eps_S: float, eps_B: float):
        self.likelihood = Sequential(
            _Fail(lam_S, lam_B, eps_S, eps_B),
            _Pass(lam_S, lam_B, eps_S, eps_B)
        )

    def __call__(self, k_before: int, k_after: int):
        if k_after > k_before:
            raise ValueError("k_after cannot exceed k_before")

        k_fail = k_before - k_after
        k_pass = k_after

        return self.likelihood(k_fail, k_pass)

    def sample(self, size=1):
        k_fail, k_pass = self.likelihood.sample(size=size)

        k_before = k_fail + k_pass
        k_after = k_pass

        return k_before, k_after

class BinnedLikelihood(LikelihoodBase):
    def __init__(self, sig_model, bg_model, bin_edges: np.ndarray):
        super(BinnedLikelihood, self).__init__()

        self.sig_model = sig_model
        self.bg_model = bg_model

        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.bin_lo = self.bin_edges[:-1]
        self.bin_hi = self.bin_edges[1:]

    def update(self, eps_S: float, eps_B: float, sig_pars, bg_pars):
        lam_S = self.sig_model.integral(self.bin_lo, self.bin_hi, *sig_pars)
        lam_B = self.bg_model.integral(self.bin_lo, self.bin_hi, *bg_pars)

        self.likelihood = Sequential(
            *(OneBin(lam_S[i], lam_B[i], eps_S, eps_B) for i in range(len(lam_S)))
        )

    def __call__(self, observed_counts: np.ndarray, eps_S, eps_B, sig_pars, bg_pars):
        self.update(eps_S, eps_B, sig_pars, bg_pars)

        args = tuple(tuple(obs) for obs in observed_counts)
        return self.likelihood(*args)

    def sample(self, eps_S, eps_B, sig_pars, bg_pars, size=1):
        self.update(eps_S, eps_B, sig_pars, bg_pars)
        return self.likelihood.sample(size=size)
