import numpy as np

from pyinfer.inference.parameters import ParameterMap
from pyinfer.inference.fit import MinuitFitter
from pyinfer.inference.feldman_cousins import FeldmanCousins


class GaussianToyProblem:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.parameter_map = ParameterMap(
            signal_names=(),
            background_names=(),
            poi="eps_S",
        )

    def nll(self, data, pars):
        return (
            0.5 * ((data - pars["eps_S"]) / self.sigma) ** 2
            + 0.5 * ((pars["eps_B"] - 0.2) / 0.1) ** 2
        )

    def sample(self, pars, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(pars["eps_S"], self.sigma, size=size)


def make_fitter():
    problem = GaussianToyProblem()

    fitter = MinuitFitter(
        problem,
        limits={
            "eps_S": (0, 1),
            "eps_B": (0, 1),
        },
    )

    return problem, fitter


def test_fc_reproducible():
    problem, fitter = make_fitter()

    start = {
        "eps_S": 0.5,
        "eps_B": 0.2,
    }

    kwargs = dict(
        problem=problem,
        fitter=fitter,
        confidence_level=0.9,
        n_toys=8,
        seed=12345,
        n_jobs=1,
    )

    a = FeldmanCousins(**kwargs).run(
        0.6,
        [0.5, 0.6, 0.7],
        start,
    )

    b = FeldmanCousins(**kwargs).run(
        0.6,
        [0.5, 0.6, 0.7],
        start,
    )

    for pa, pb in zip(a.points, b.points):
        np.testing.assert_array_equal(pa.q_toys, pb.q_toys)


def test_fc_serial_parallel_identical():
    problem, fitter = make_fitter()

    start = {
        "eps_S": 0.5,
        "eps_B": 0.2,
    }

    serial = FeldmanCousins(
        problem,
        fitter,
        confidence_level=0.9,
        n_toys=8,
        seed=12345,
        n_jobs=1,
    ).run(
        0.6,
        [0.5, 0.6, 0.7],
        start,
    )

    parallel = FeldmanCousins(
        problem,
        fitter,
        confidence_level=0.9,
        n_toys=8,
        seed=12345,
        n_jobs=2,
    ).run(
        0.6,
        [0.5, 0.6, 0.7],
        start,
    )

    assert serial.seed_entropy == parallel.seed_entropy

    for ps, pp in zip(serial.points, parallel.points):
        np.testing.assert_array_equal(ps.q_toys, pp.q_toys)
        assert ps.q_obs == pp.q_obs
        assert ps.q_crit == pp.q_crit
        assert ps.p_value == pp.p_value
        assert ps.accepted == pp.accepted


def test_fc_different_seed_changes_toys():
    problem, fitter = make_fitter()
    start = {"eps_S": 0.5, "eps_B": 0.2}

    a = FeldmanCousins(
        problem,
        fitter,
        n_toys=8,
        seed=123,
        n_jobs=1,
    ).run(
        0.6,
        [0.6],
        start,
    )

    b = FeldmanCousins(
        problem,
        fitter,
        n_toys=8,
        seed=456,
        n_jobs=1,
    ).run(
        0.6,
        [0.6],
        start,
    )

    assert not np.array_equal(
        a.points[0].q_toys,
        b.points[0].q_toys,
    )
