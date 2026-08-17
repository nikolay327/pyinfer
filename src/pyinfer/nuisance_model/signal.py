import numpy as np
import scipy.stats as stats

def Gauss(x: np.ndarray, A, mu, sig):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

