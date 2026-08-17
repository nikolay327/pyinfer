import numpy as np

from pyinfer.inference.parameters import ParameterMap
from pyinfer.inference.fit import MinuitFitter
from pyinfer.inference.profile import profile_likelihood_ratio, profile_scan


class QuadraticProblem:
    def __init__(self):
        self.parameter_map = ParameterMap(
            signal_names=("x",),
            background_names=(),
            poi="eps_S",
        )

    def nll(self, data, pars):
        return (
            0.5 * ((pars["eps_S"] - 0.7) / 0.1) ** 2
            + 0.5 * ((pars["eps_B"] - 0.2) / 0.1) ** 2
            + 0.5 * ((pars["x"] - 1.5) / 0.2) ** 2
        )


def make_fitter():
    problem = QuadraticProblem()

    limits = {
        "eps_S": (0, 1),
        "eps_B": (0, 1),
        "x": (None, None),
    }

    return MinuitFitter(problem, limits=limits)


def test_global_fit():
    fitter = make_fitter()

    result = fitter.fit(
        None,
        {
            "eps_S": 0.4,
            "eps_B": 0.5,
            "x": 0.0,
        },
    )

    assert result.valid
    np.testing.assert_allclose(result.values["eps_S"], 0.7, atol=1e-3)
    np.testing.assert_allclose(result.values["eps_B"], 0.2, atol=1e-3)
    np.testing.assert_allclose(result.values["x"], 1.5, atol=1e-3)


def test_fixed_parameter_fit():
    fitter = make_fitter()

    result = fitter.fit(
        None,
        {
            "eps_S": 0.4,
            "eps_B": 0.5,
            "x": 0.0,
        },
        fixed={"eps_S": 0.5},
    )

    assert result.valid
    assert result.values["eps_S"] == 0.5


def test_profile_likelihood_ratio():
    fitter = make_fitter()
    start = {"eps_S": 0.4, "eps_B": 0.5, "x": 0.0}

    result = profile_likelihood_ratio(
        fitter,
        None,
        poi_value=0.5,
        start=start,
    )

    assert result.valid

    # delta NLL = 0.5 * ((0.5 - 0.7) / 0.1)^2 = 2
    # q = 2 * delta NLL = 4
    np.testing.assert_allclose(result.q, 4.0, atol=1e-4)


def test_profile_scan_uses_one_global_fit():
    fitter = make_fitter()
    start = {"eps_S": 0.4, "eps_B": 0.5, "x": 0.0}

    results = profile_scan(
        fitter,
        None,
        [0.5, 0.6, 0.7, 0.8],
        start,
    )

    global_fit = results[0].global_fit

    assert all(result.global_fit is global_fit for result in results)
    assert all(result.valid for result in results)
