from .base import ModelBase
from .signal import Gauss, GaussEMGLeft
from .background import Polynomial, PolyStep


class GaussEMGLeftPolyStep(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        self.signal = GaussEMGLeft()
        self.background = PolyStep(x_ref, x_scale)

    def __call__(self, x, A, mu, sig, f_tail, tau, A_step, b0, *hs):
        return (
            self.signal(x, A, mu, sig, f_tail, tau)
            + self.background(x, mu, sig, A_step, b0, *hs)
        )

    def integral(self, x_lo, x_hi, A, mu, sig, f_tail, tau, A_step, b0, *hs):
        return (
            self.signal.integral(x_lo, x_hi, A, mu, sig, f_tail, tau)
            + self.background.integral(
                x_lo, x_hi, mu, sig, A_step, b0, *hs
            )
        )


class GaussEMGLeftPolynomial(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        self.signal = GaussEMGLeft()
        self.background = Polynomial(x_ref, x_scale)

    def __call__(self, x, A, mu, sig, f_tail, tau, b0, *hs):
        return (
            self.signal(x, A, mu, sig, f_tail, tau)
            + self.background(x, b0, *hs)
        )

    def integral(self, x_lo, x_hi, A, mu, sig, f_tail, tau, b0, *hs):
        return (
            self.signal.integral(x_lo, x_hi, A, mu, sig, f_tail, tau)
            + self.background.integral(x_lo, x_hi, b0, *hs)
        )


class GaussPolyStep(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        self.signal = Gauss()
        self.background = PolyStep(x_ref, x_scale)

    def __call__(self, x, A, mu, sig, A_step, b0, *hs):
        return (
            self.signal(x, A, mu, sig)
            + self.background(x, mu, sig, A_step, b0, *hs)
        )

    def integral(self, x_lo, x_hi, A, mu, sig, A_step, b0, *hs):
        return (
            self.signal.integral(x_lo, x_hi, A, mu, sig)
            + self.background.integral(
                x_lo, x_hi, mu, sig, A_step, b0, *hs
            )
        )


class GaussPolynomial(ModelBase):
    def __init__(self, x_ref=0.0, x_scale=1.0):
        self.signal = Gauss()
        self.background = Polynomial(x_ref, x_scale)

    def __call__(self, x, A, mu, sig, b0, *hs):
        return (
            self.signal(x, A, mu, sig)
            + self.background(x, b0, *hs)
        )

    def integral(self, x_lo, x_hi, A, mu, sig, b0, *hs):
        return (
            self.signal.integral(x_lo, x_hi, A, mu, sig)
            + self.background.integral(x_lo, x_hi, b0, *hs)
        )
