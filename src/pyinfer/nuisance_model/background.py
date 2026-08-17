import numpy as np
from scipy.special import erfc

from .base import ModelBase


class Polynomial(ModelBase):
    def __call__(self, x, *coeffs):
        return np.polynomial.polynomial.polyval(x, coeffs)

    def integral(self, x_lo, x_hi, *coeffs):
        int_coeffs = np.polynomial.polynomial.polyint(coeffs)
        return np.polynomial.polynomial.polyval(x_hi, int_coeffs) - np.polynomial.polynomial.polyval(x_lo, int_coeffs)


class Step(ModelBase):
    def __call__(self, x, A, mu, sig):
        return 0.5 * A * erfc((x - mu) / (np.sqrt(2.0) * sig))

    def integral(self, x_lo, x_hi, A, mu, sig):
        return A * (self._primitive(x_hi, mu, sig) - self._primitive(x_lo, mu, sig))

    @staticmethod
    def _primitive(x, mu, sig):
        z = (x - mu) / (np.sqrt(2.0) * sig)
        return sig / np.sqrt(2.0) * (z * erfc(z) - np.exp(-z**2) / np.sqrt(np.pi))


class PolyStep(ModelBase):
    def __init__(self):
        self.poly = Polynomial()
        self.step = Step()

    def __call__(self, x, mu, sig, h_step, *coeffs):
        poly = self.poly(x, *coeffs)
        A_step = h_step * self.poly(mu, *coeffs)
        return poly + self.step(x, A_step, mu, sig)

    def integral(self, x_lo, x_hi, mu, sig, h_step, *coeffs):
        poly_int = self.poly.integral(x_lo, x_hi, *coeffs)
        A_step = h_step * self.poly(mu, *coeffs)
        return poly_int + self.step.integral(x_lo, x_hi, A_step, mu, sig)
