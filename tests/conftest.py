import numpy as np
import pytest

from pyinfer.inference.parameters import get_parameter_map
from pyinfer.inference.problem import InferenceProblem
from pyinfer.likelihood.binned_gamma import BinnedLikelihood
from pyinfer.nuisance_model.signal import Gauss
from pyinfer.nuisance_model.background import Polynomial


@pytest.fixture
def gamma_problem():
    bin_edges = np.linspace(-4, 4, 9)

    likelihood = BinnedLikelihood(
        Gauss(),
        Polynomial(),
        bin_edges,
    )

    parameter_map = get_parameter_map(
        "GaussPolynomial",
        degree=0,
    )

    problem = InferenceProblem(
        likelihood,
        parameter_map,
    )

    pars = {
        "eps_S": 0.75,
        "eps_B": 0.15,
        "A": 300.0,
        "mu": 0.2,
        "sig": 0.8,
        "c0": 25.0,
    }

    return problem, pars, bin_edges
