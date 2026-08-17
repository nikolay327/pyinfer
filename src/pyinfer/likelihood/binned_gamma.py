import numpy as np

from .base import LikelihoodBase, Sequential
from .poisson import Poisson


class _Fail(LikelihoodBase):
    def __init__(self, lam_S, lam_B, eps_S, eps_B):
        super(_Fail, self).__init__()
        self.likelihood = Poisson((1 - eps_S) * lam_S + (1 - eps_B) * lam_B)

    def __call__(self, k):
        return self.likelihood(k)

    def sample(self, size=1, rng=None):
        return self.likelihood.sample(size=size, rng=rng)


class _Pass(LikelihoodBase):
    def __init__(self, lam_S, lam_B, eps_S, eps_B):
        super(_Pass, self).__init__()
        self.likelihood = Poisson(eps_S * lam_S + eps_B * lam_B)

    def __call__(self, k):
        return self.likelihood(k)

    def sample(self, size=1, rng=None):
        return self.likelihood.sample(size=size, rng=rng)


class OneBin(LikelihoodBase):
    def __init__(self, lam_S, lam_B, eps_S, eps_B):
        super(OneBin, self).__init__()
        self.likelihood = Sequential(
            _Fail(lam_S, lam_B, eps_S, eps_B),
            _Pass(lam_S, lam_B, eps_S, eps_B)
        )

    def __call__(self, k_before, k_after):
        if k_before < 0 or k_after < 0:
            raise ValueError("Counts must be non-negative")
        if k_after > k_before:
            raise ValueError("k_after cannot exceed k_before")

        return self.likelihood(k_before - k_after, k_after)

    def sample(self, size=1, rng=None):
        k_fail, k_pass = self.likelihood.sample(size=size, rng=rng)
        return k_fail + k_pass, k_pass

class BinnedLikelihood(LikelihoodBase):
    def __init__(self, sig_model, bg_model, bin_edges: np.ndarray):
        super(BinnedLikelihood, self).__init__()

        self.sig_model = sig_model
        self.bg_model = bg_model
        self.bin_edges = np.asarray(bin_edges, dtype=float)

        if self.bin_edges.ndim != 1 or len(self.bin_edges) < 2:
            raise ValueError("bin_edges must be a one-dimensional array")
        if np.any(~np.isfinite(self.bin_edges)) or np.any(np.diff(self.bin_edges) <= 0):
            raise ValueError("bin_edges must be finite and strictly increasing")

        self.bin_lo = self.bin_edges[:-1]
        self.bin_hi = self.bin_edges[1:]

    def _make_likelihood(self, eps_S, eps_B, sig_pars, bg_pars):
        if not 0 <= eps_S <= 1 or not 0 <= eps_B <= 1:
            raise ValueError("Efficiencies must be between 0 and 1")

        lam_S = np.asarray(self.sig_model.integral(self.bin_lo, self.bin_hi, *sig_pars))
        lam_B = np.asarray(self.bg_model.integral(self.bin_lo, self.bin_hi, *bg_pars))

        if lam_S.shape != self.bin_lo.shape or lam_B.shape != self.bin_lo.shape:
            raise ValueError("Model integrals have incorrect shape")

        if np.any(~np.isfinite(lam_S)) or np.any(lam_S < 0):
            raise ValueError("Signal expectations must be finite and non-negative")

        if np.any(~np.isfinite(lam_B)) or np.any(lam_B < 0):
            raise ValueError("Background expectations must be finite and non-negative")

        return Sequential(
            *(OneBin(lam_S[i], lam_B[i], eps_S, eps_B) for i in range(len(lam_S)))
        )

    def __call__(self, observed_counts, eps_S, eps_B, sig_pars, bg_pars):
        data = np.asarray(observed_counts)

        if data.shape != (len(self.bin_lo), 2):
            raise ValueError("observed_counts must have shape (n_bins, 2)")
        if np.any(~np.isfinite(data)) or np.any(data < 0):
            raise ValueError("Observed counts must be finite and non-negative")
        if np.any(data != np.floor(data)):
            raise ValueError("Observed counts must be integers")
        if np.any(data[:, 1] > data[:, 0]):
            raise ValueError("k_after cannot exceed k_before")

        likelihood = self._make_likelihood(eps_S, eps_B, sig_pars, bg_pars)
        return likelihood(*(tuple(obs) for obs in data))

    def sample(self, eps_S, eps_B, sig_pars, bg_pars, size=1, rng=None):
        likelihood = self._make_likelihood(eps_S, eps_B, sig_pars, bg_pars)
        samples = likelihood.sample(size=size, rng=rng)

        before = np.stack([sample[0] for sample in samples], axis=1)
        after = np.stack([sample[1] for sample in samples], axis=1)

        return np.stack((before, after), axis=-1)

