import numpy as np
import pytest

from pyinfer.inference.initialize import get_initial_params
from pyinfer.nuisance_model.sig_plus_bg import GaussPolynomial


def test_initializer():
    edges = np.linspace(-5, 5, 41)
    model = GaussPolynomial(x_ref=0.0, x_scale=5.0)

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

    assert set(params) == {"A", "mu", "sig", "b0"}
    assert set(("eps_S", "eps_B")).issubset(limits)
    assert all(np.isfinite(value) for value in params.values())
    assert result.success
    assert np.isfinite(result.fun)
    assert result.nit == 0
    assert result.nfev == 1


def test_initializer_is_deterministic():
    edges = np.linspace(-5, 5, 41)
    model = GaussPolynomial(x_ref=0.0, x_scale=5.0)

    expected = model.integral(
        edges[:-1],
        edges[1:],
        500.0,
        0.3,
        0.7,
        10.0,
    )

    counts = np.rint(expected).astype(int)

    a, _, _ = get_initial_params(
        "GaussPolynomial",
        edges,
        counts,
        degree=0,
    )

    b, _, _ = get_initial_params(
        "GaussPolynomial",
        edges,
        counts,
        degree=0,
    )

    assert a == b


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

def test_step_initializer_returns_independent_amplitude():
    from pyinfer.nuisance_model.sig_plus_bg import GaussPolyStep

    edges = np.linspace(-5, 5, 81)

    model = GaussPolyStep(
        x_ref=0.0,
        x_scale=5.0,
    )

    expected = model.integral(
        edges[:-1],
        edges[1:],
        300.0,  # A
        0.2,    # mu
        0.7,    # sig
        12.0,   # A_step
        2.0,    # b0
        -0.1,   # h1
    )

    counts = np.rint(expected).astype(int)

    params, limits, result = get_initial_params(
        "GaussPolyStep",
        edges,
        counts,
        degree=1,
    )

    assert "A_step" in params
    assert "h_step" not in params
    assert np.isfinite(params["A_step"])
    assert params["A_step"] >= 0
    assert limits["A_step"] == (0, None)
    assert result.success
