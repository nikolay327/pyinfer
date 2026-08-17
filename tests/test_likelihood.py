import numpy as np
import pytest
import scipy.stats as stats

from pyinfer.likelihood.poisson import Poisson
from pyinfer.likelihood.binned_gamma import OneBin


def test_poisson_log_likelihood():
    likelihood = Poisson(4.2)
    np.testing.assert_allclose(
        likelihood(3),
        stats.poisson.logpmf(3, 4.2),
    )


def test_poisson_sampling_reproducible():
    likelihood = Poisson(4.2)

    a = likelihood.sample(size=100, rng=np.random.default_rng(123))
    b = likelihood.sample(size=100, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(a, b)


def test_one_bin_likelihood():
    lam_S, lam_B = 10.0, 20.0
    eps_S, eps_B = 0.8, 0.2
    before, after = 15, 8

    likelihood = OneBin(lam_S, lam_B, eps_S, eps_B)

    lam_fail = (1 - eps_S) * lam_S + (1 - eps_B) * lam_B
    lam_pass = eps_S * lam_S + eps_B * lam_B

    expected = (
        stats.poisson.logpmf(before - after, lam_fail)
        + stats.poisson.logpmf(after, lam_pass)
    )

    np.testing.assert_allclose(likelihood(before, after), expected)


def test_binned_likelihood_matches_manual(gamma_problem):
    problem, pars, _ = gamma_problem
    likelihood = problem.likelihood

    data = problem.sample(
        pars,
        size=1,
        rng=np.random.default_rng(123),
    )[0]

    lam_S = likelihood.sig_model.integral(
        likelihood.bin_lo,
        likelihood.bin_hi,
        pars["A"],
        pars["mu"],
        pars["sig"],
    )

    lam_B = likelihood.bg_model.integral(
        likelihood.bin_lo,
        likelihood.bin_hi,
        pars["c0"],
    )

    lam_fail = (1 - pars["eps_S"]) * lam_S + (1 - pars["eps_B"]) * lam_B
    lam_pass = pars["eps_S"] * lam_S + pars["eps_B"] * lam_B

    manual = np.sum(
        stats.poisson.logpmf(data[:, 0] - data[:, 1], lam_fail)
        + stats.poisson.logpmf(data[:, 1], lam_pass)
    )

    np.testing.assert_allclose(problem.log_likelihood(data, pars), manual)


def test_binned_sample_shape(gamma_problem):
    problem, pars, _ = gamma_problem

    toys = problem.sample(
        pars,
        size=7,
        rng=np.random.default_rng(123),
    )

    assert toys.shape == (7, 8, 2)
    assert np.all(toys[:, :, 1] <= toys[:, :, 0])


def test_binned_sampling_reproducible(gamma_problem):
    problem, pars, _ = gamma_problem

    a = problem.sample(pars, size=10, rng=np.random.default_rng(123))
    b = problem.sample(pars, size=10, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(a, b)


def test_after_cannot_exceed_before(gamma_problem):
    problem, pars, _ = gamma_problem
    data = np.zeros((8, 2), dtype=int)
    data[0] = (2, 3)

    with pytest.raises(ValueError):
        problem.log_likelihood(data, pars)
