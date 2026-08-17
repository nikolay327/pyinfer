import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from ..nuisance_model.sig_plus_bg import (
    GaussEMGLeftPolyStep,
    GaussEMGLeftPolynomial,
    GaussPolyStep,
    GaussPolynomial,
)


MODELS = {
    "GaussEMGLeftPolyStep": GaussEMGLeftPolyStep,
    "GaussEMGLeftPolynomial": GaussEMGLeftPolynomial,
    "GaussPolyStep": GaussPolyStep,
    "GaussPolynomial": GaussPolynomial
}


def get_initial_params(model_name, bin_edges, bin_contents, degree=1):
    edges = np.asarray(bin_edges, dtype=float)
    y = np.asarray(bin_contents, dtype=float)

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if edges.ndim != 1 or y.ndim != 1:
        raise ValueError("bin_edges and bin_contents must be one-dimensional")

    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a non-negative integer")

    if len(edges) != len(y) + 1:
        raise ValueError("bin_edges must have length len(bin_contents) + 1")

    if len(y) < 3:
        raise ValueError("At least three bins are required")

    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be finite and strictly increasing")

    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("bin_contents must be finite and non-negative")

    if np.any(y != np.floor(y)):
        raise ValueError("bin_contents must contain integer counts")

    if degree >= len(y):
        raise ValueError("degree must be smaller than the number of bins")

    x_lo = edges[:-1]
    x_hi = edges[1:]
    x = 0.5 * (x_lo + x_hi)
    widths = x_hi - x_lo

    xmin, xmax = edges[0], edges[-1]
    span = xmax - xmin
    min_width = np.min(widths)

    n_edge = max(degree + 1, int(0.2 * len(x)))
    edge_mask = np.zeros(len(x), dtype=bool)
    edge_mask[:n_edge] = True
    edge_mask[-n_edge:] = True

    y_density = y / widths
    coeffs0 = np.polynomial.polynomial.polyfit(x[edge_mask], y_density[edge_mask], degree)

    bg0 = np.polynomial.polynomial.polyval(x, coeffs0)
    excess = np.clip(y_density - bg0, 0, None)

    mu0 = x[np.argmax(excess)]

    local = np.abs(x - mu0) < 0.1 * span
    weights = excess[local] * widths[local]

    if np.sum(weights) > 0:
        sig0 = np.sqrt(np.sum(weights * (x[local] - mu0) ** 2) / np.sum(weights))
    else:
        sig0 = 2 * min_width

    sig0 = np.clip(sig0, min_width / 2, span / 5)
    A0 = np.sum(excess * widths)

    if model_name == "GaussEMGLeftPolyStep":
        p0 = [A0, mu0, sig0, 0.1, sig0, 0.1, *coeffs0]
        names = ["A", "mu", "sig", "f_tail", "tau", "h_step"] + [f"c{i}" for i in range(degree + 1)]
        bounds = [(0, None), (xmin, xmax), (min_width / 10, span), (0, 1), (min_width / 10, span), (-0.9, 1)] + [(None, None)] * (degree + 1)

    elif model_name == "GaussEMGLeftPolynomial":
        p0 = [A0, mu0, sig0, 0.1, sig0, *coeffs0]
        names = ["A", "mu", "sig", "f_tail", "tau"] + [f"c{i}" for i in range(degree + 1)]
        bounds = [(0, None), (xmin, xmax), (min_width / 10, span), (0, 1), (min_width / 10, span)] + [(None, None)] * (degree + 1)

    elif model_name == "GaussPolyStep":
        p0 = [A0, mu0, sig0, 0.1, *coeffs0]
        names = ["A", "mu", "sig", "h_step"] + [f"c{i}" for i in range(degree + 1)]
        bounds = [(0, None), (xmin, xmax), (min_width / 10, span), (-0.9, 1)] + [(None, None)] * (degree + 1)

    else:
        p0 = [A0, mu0, sig0, *coeffs0]
        names = ["A", "mu", "sig"] + [f"c{i}" for i in range(degree + 1)]
        bounds = [(0, None), (xmin, xmax), (min_width / 10, span)] + [(None, None)] * (degree + 1)

    model = MODELS[model_name]()

    def nll(pars):
        expected = model.integral(x_lo, x_hi, *pars)

        if np.any(~np.isfinite(expected)) or np.any(expected < 0):
            return np.inf

        return -np.sum(stats.poisson.logpmf(y, expected))

    result = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)

    limits = {
        "eps_S": (0, 1),
        "eps_B": (0, 1),
        **dict(zip(names, bounds)),
    }

    if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
        values = result.x
    else:
        values = p0

    params = dict(zip(names, values))
    return params, limits, result
