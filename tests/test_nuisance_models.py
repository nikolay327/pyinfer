import numpy as np
import pytest
from scipy.integrate import quad

from pyinfer.nuisance_model.signal import Gauss, EMG, GaussEMGLeft
from pyinfer.nuisance_model.background import Polynomial, Step, PolyStep


def test_gauss_area():
    model = Gauss()
    value = model.integral(-100, 100, 12.3, 1.0, 0.8)
    np.testing.assert_allclose(value, 12.3, rtol=1e-12)


def test_emg_left_area():
    model = EMG(tail="left")
    value = model.integral(-100, 100, 12.3, 1.0, 0.8, 1.2)
    np.testing.assert_allclose(value, 12.3, rtol=1e-10)


def test_gauss_emg_area():
    model = GaussEMGLeft()
    value = model.integral(-100, 100, 20.0, 0.0, 0.8, 0.25, 1.1)
    np.testing.assert_allclose(value, 20.0, rtol=1e-10)


def test_polynomial_integral():
    model = Polynomial(x_ref=0.5, x_scale=2.0)
    pars = (4.0, -0.2, 0.1)
    lo, hi = -1.0, 2.0

    exact = model.integral(lo, hi, *pars)
    numeric = quad(lambda x: model(x, *pars), lo, hi)[0]

    np.testing.assert_allclose(exact, numeric, rtol=1e-12)


def test_polynomial_scaled_coordinates():
    model = Polynomial(x_ref=2600.0, x_scale=50.0)

    x = np.array([
        2550.0,
        2600.0,
        2650.0,
    ])

    y = model(
        x,
        100.0,
        -0.1,
    )

    np.testing.assert_allclose(
        y,
        [110.0, 100.0, 90.0],
    )


def test_invalid_polynomial_scale():
    with pytest.raises(ValueError):
        Polynomial(
            x_ref=0.0,
            x_scale=0.0,
        )


def test_step_integral():
    model = Step()
    pars = (3.0, 0.2, 0.8)
    lo, hi = -2.0, 3.0

    exact = model.integral(lo, hi, *pars)
    numeric = quad(lambda x: model(x, *pars), lo, hi)[0]

    np.testing.assert_allclose(exact, numeric, rtol=1e-10)


def test_polystep_integral():
    model = PolyStep(
        x_ref=0.0,
        x_scale=2.0,
    )

    pars = (
        0.2,   # mu
        0.8,   # sig
        6.0,   # A_step
        4.0,   # b0
        0.1,   # h1
    )

    lo, hi = -2.0, 3.0

    exact = model.integral(
        lo,
        hi,
        *pars,
    )

    numeric = quad(
        lambda x: model(x, *pars),
        lo,
        hi,
    )[0]

    np.testing.assert_allclose(
        exact,
        numeric,
        rtol=1e-10,
    )


def test_polystep_independent_step_amplitude():
    model = PolyStep(
        x_ref=0.0,
        x_scale=1.0,
    )

    left = model(
        -10.0,
        0.0,
        0.5,
        7.0,
        2.0,
    )

    right = model(
        10.0,
        0.0,
        0.5,
        7.0,
        2.0,
    )

    np.testing.assert_allclose(left, 9.0, rtol=1e-10)
    np.testing.assert_allclose(right, 2.0, rtol=1e-10)


def test_invalid_emg_tail():
    with pytest.raises(ValueError):
        EMG(tail="banana")
