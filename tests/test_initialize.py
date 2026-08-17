import numpy as np
import pytest

from pyinfer.inference.initialize import get_initial_params
from pyinfer.nuisance_model.sig_plus_bg import GaussPolynomial


def test_initializer():
    edges = np.linspace(-5, 5, 41)
    model = GaussPolynomial()

    expected = model.integral(
        edges[:-1],
        edges[1:],
        500.0,
        0.3,
        0.7,
        10.0,
    )

    counts = np.rint(expected).astype(int)

    params, limits, result = get_initial_params(
        "GaussPolynomial",
        edges,
        counts,
        degree=0,
    )

    assert set(params) == {"A", "mu", "sig", "c0"}
    assert set(("eps_S", "eps_B")).issubset(limits)
    assert all(np.isfinite(value) for value in params.values())
    assert np.isfinite(result.fun)


def test_initializer_rejects_non_monotonic_edges():
    edges = np.array([0.0, 1.0, 0.5, 2.0])
    counts = np.array([10, 10, 10])

    with pytest.raises(ValueError):
        get_initial_params(
            "GaussPolynomial",
            edges,
            counts,
            degree=0,
        )


def test_initializer_rejects_negative_counts():
    edges = np.arange(5.0)
    counts = np.array([10, -1, 10, 10])

    with pytest.raises(ValueError):
        get_initial_params(
            "GaussPolynomial",
            edges,
            counts,
            degree=0,
        )
