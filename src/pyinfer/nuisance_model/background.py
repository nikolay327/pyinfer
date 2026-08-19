import numpy as np
from scipy.special import erfc

from .base import ModelBase


class Polynomial(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        if not np.isfinite(x_ref):
            raise ValueError("x_ref must be finite")
        if not np.isfinite(x_scale) or x_scale <= 0:
            raise ValueError("x_scale must be finite and positive")

        self.x_ref = x_ref
        self.x_scale = x_scale

    def __call__(self, x, b0, *hs):
        z = (np.asarray(x) - self.x_ref) / self.x_scale
        return b0 * np.polynomial.polynomial.polyval(z, (1.0, *hs))

    def integral(self, x_lo, x_hi, b0, *hs):
        z_lo = (np.asarray(x_lo) - self.x_ref) / self.x_scale
        z_hi = (np.asarray(x_hi) - self.x_ref) / self.x_scale
        coeffs = np.polynomial.polynomial.polyint((1.0, *hs))

        return b0 * self.x_scale * (
            np.polynomial.polynomial.polyval(z_hi, coeffs)
            - np.polynomial.polynomial.polyval(z_lo, coeffs)
        )


class Step(ModelBase):
    def __call__(self, x, A, mu, sig):
        return 0.5 * A * erfc((x - mu) / (np.sqrt(2.0) * sig))

    def integral(self, x_lo, x_hi, A, mu, sig):
        return A * (
            self._primitive(x_hi, mu, sig)
            - self._primitive(x_lo, mu, sig)
        )

    @staticmethod
    def _primitive(x, mu, sig):
        z = (x - mu) / (np.sqrt(2.0) * sig)
        return sig / np.sqrt(2.0) * (
            z * erfc(z)
            - np.exp(-z**2) / np.sqrt(np.pi)
        )


class PolyStep(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        self.poly = Polynomial(x_ref, x_scale)
        self.step = Step()

    def __call__(self, x, mu, sig, A_step, b0, *hs):
        return (
            self.poly(x, b0, *hs)
            + self.step(x, A_step, mu, sig)
        )

    def integral(self, x_lo, x_hi, mu, sig, A_step, b0, *hs):
        return (
            self.poly.integral(x_lo, x_hi, b0, *hs)
            + self.step.integral(x_lo, x_hi, A_step, mu, sig)
        )
