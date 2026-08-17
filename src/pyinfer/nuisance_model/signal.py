import scipy.stats as stats
from .base import ModelBase

class Gauss(ModelBase):
    def __call__(self, x, A, mu, sig):
        return A * stats.norm.pdf(x, loc=mu, scale=sig)

    def integral(self, x_lo, x_hi, A, mu, sig):
        return A * (
            stats.norm.cdf(x_hi, loc=mu, scale=sig)
            - stats.norm.cdf(x_lo, loc=mu, scale=sig)
        )
    
class EMG(ModelBase):
    def __init__(self, tail="left"):
        if tail not in ("left", "right"):
            raise ValueError("tail must be either 'left' or 'right'")
        self.tail = tail

    def __call__(self, x, A, mu, sig, tau):
        K = tau / sig

        if self.tail == "right":
            return A * stats.exponnorm.pdf(x, K=K, loc=mu, scale=sig)

        return A * stats.exponnorm.pdf(-x, K=K, loc=-mu, scale=sig)

    def integral(self, x_lo, x_hi, A, mu, sig, tau):
        K = tau / sig

        if self.tail == "right":
            return A * (
                stats.exponnorm.cdf(x_hi, K=K, loc=mu, scale=sig)
                - stats.exponnorm.cdf(x_lo, K=K, loc=mu, scale=sig)
            )

        return A * (
            stats.exponnorm.cdf(-x_lo, K=K, loc=-mu, scale=sig)
            - stats.exponnorm.cdf(-x_hi, K=K, loc=-mu, scale=sig)
        )

class GaussEMGLeft(ModelBase):
    def __init__(self):
        self.gauss = Gauss()
        self.emg = EMG(tail="left")

    def __call__(self, x, A, mu, sig, f_tail, tau):
        return (
            self.gauss(x, A * (1 - f_tail), mu, sig)
            + self.emg(x, A * f_tail, mu, sig, tau)
        )

    def integral(self, x_lo, x_hi, A, mu, sig, f_tail, tau):
        return (
            self.gauss.integral(x_lo, x_hi, A * (1 - f_tail), mu, sig)
            + self.emg.integral(x_lo, x_hi, A * f_tail, mu, sig, tau)
        )
