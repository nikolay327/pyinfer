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
    model = Polynomial()
    coeffs = (2.0, -0.5, 0.1)
    lo, hi = -1.0, 2.0

    exact = model.integral(lo, hi, *coeffs)
    numeric = quad(lambda x: model(x, *coeffs), lo, hi)[0]

    np.testing.assert_allclose(exact, numeric, rtol=1e-12)


def test_step_integral():
    model = Step()
    pars = (3.0, 0.2, 0.8)
    lo, hi = -2.0, 3.0

    exact = model.integral(lo, hi, *pars)
    numeric = quad(lambda x: model(x, *pars), lo, hi)[0]

    np.testing.assert_allclose(exact, numeric, rtol=1e-10)


def test_polystep_integral():
    model = PolyStep()
    pars = (0.2, 0.8, 0.3, 4.0, 0.1)
    lo, hi = -2.0, 3.0

    exact = model.integral(lo, hi, *pars)
    numeric = quad(lambda x: model(x, *pars), lo, hi)[0]

    np.testing.assert_allclose(exact, numeric, rtol=1e-10)


def test_invalid_emg_tail():
    with pytest.raises(ValueError):
        EMG(tail="banana")
