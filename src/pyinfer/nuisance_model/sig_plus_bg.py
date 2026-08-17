from .base import ModelBase
from .signal import Gauss, GaussEMGLeft
from .background import Polynomial, PolyStep


class GaussEMGLeftPolyStep(ModelBase):
    def __init__(self):
        self.signal = GaussEMGLeft()
        self.background = PolyStep()

    def __call__(self, x, A, mu, sig, f_tail, tau, h_step, *coeffs):
        return self.signal(x, A, mu, sig, f_tail, tau) + self.background(x, mu, sig, h_step, *coeffs)

    def integral(self, x_lo, x_hi, A, mu, sig, f_tail, tau, h_step, *coeffs):
        return self.signal.integral(x_lo, x_hi, A, mu, sig, f_tail, tau) + self.background.integral(x_lo, x_hi, mu, sig, h_step, *coeffs)


class GaussEMGLeftPolynomial(ModelBase):
    def __init__(self):
        self.signal = GaussEMGLeft()
        self.background = Polynomial()

    def __call__(self, x, A, mu, sig, f_tail, tau, *coeffs):
        return self.signal(x, A, mu, sig, f_tail, tau) + self.background(x, *coeffs)

    def integral(self, x_lo, x_hi, A, mu, sig, f_tail, tau, *coeffs):
        return self.signal.integral(x_lo, x_hi, A, mu, sig, f_tail, tau) + self.background.integral(x_lo, x_hi, *coeffs)


class GaussPolyStep(ModelBase):
    def __init__(self):
        self.signal = Gauss()
        self.background = PolyStep()

    def __call__(self, x, A, mu, sig, h_step, *coeffs):
        return self.signal(x, A, mu, sig) + self.background(x, mu, sig, h_step, *coeffs)

    def integral(self, x_lo, x_hi, A, mu, sig, h_step, *coeffs):
        return self.signal.integral(x_lo, x_hi, A, mu, sig) + self.background.integral(x_lo, x_hi, mu, sig, h_step, *coeffs)


class GaussPolynomial(ModelBase):
    def __init__(self):
        self.signal = Gauss()
        self.background = Polynomial()

    def __call__(self, x, A, mu, sig, *coeffs):
        return self.signal(x, A, mu, sig) + self.background(x, *coeffs)

    def integral(self, x_lo, x_hi, A, mu, sig, *coeffs):
        return self.signal.integral(x_lo, x_hi, A, mu, sig) + self.background.integral(x_lo, x_hi, *coeffs)
