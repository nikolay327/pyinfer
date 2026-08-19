import numpy as np
import pytest

import pyinfer
from pyinfer import GammaLineAnalysis, GammaLineConfig, GammaLineProblemBuilder
from pyinfer.nuisance_model.background import Polynomial
from pyinfer.nuisance_model.signal import Gauss


@pytest.fixture
def api_data():
    edges = np.linspace(-4.0, 4.0, 17)

    signal = Gauss()
    background = Polynomial()

    lam_S = signal.integral(
        edges[:-1],
        edges[1:],
        500.0,
        0.2,
        0.75,
    )

    lam_B = background.integral(
        edges[:-1],
        edges[1:],
        20.0,
    )

    eps_S = 0.8
    eps_B = 0.15

    rng = np.random.default_rng(12345)

    fail = rng.poisson(
        (1 - eps_S) * lam_S
        + (1 - eps_B) * lam_B
    )

    passed = rng.poisson(
        eps_S * lam_S
        + eps_B * lam_B
    )

    before = fail + passed
    after = passed

    return edges, before, after


@pytest.fixture
def analysis(api_data):
    edges, before, after = api_data

    config = GammaLineConfig(
        model="GaussPolynomial",
        degree=0,
    )

    return GammaLineAnalysis(
        bin_edges=edges,
        before=before,
        after=after,
        config=config,
    )


def test_public_imports():
    assert pyinfer.GammaLineAnalysis is GammaLineAnalysis
    assert pyinfer.GammaLineConfig is GammaLineConfig
    assert pyinfer.GammaLineProblemBuilder is GammaLineProblemBuilder


def test_config():
    config = GammaLineConfig(
        model="GaussPolynomial",
        degree=0,
    )

    assert config.model == "GaussPolynomial"
    assert config.degree == 0


def test_invalid_config():
    with pytest.raises(ValueError):
        GammaLineConfig(model="UnknownModel")

    with pytest.raises(ValueError):
        GammaLineConfig(
            model="GaussPolynomial",
            degree=-1,
        )


def test_analysis_construction(analysis):
    assert analysis.before.ndim == 1
    assert analysis.after.ndim == 1

    assert analysis.data.shape == (
        len(analysis.before),
        2,
    )

    np.testing.assert_array_equal(
        analysis.data[:, 0],
        analysis.before,
    )

    np.testing.assert_array_equal(
        analysis.data[:, 1],
        analysis.after,
    )

    assert np.all(
        analysis.after <= analysis.before
    )


def test_analysis_rejects_invalid_counts(api_data):
    edges, before, after = api_data

    bad_after = after.copy()
    bad_after[0] = before[0] + 1

    with pytest.raises(ValueError):
        GammaLineAnalysis(
            edges,
            before,
            bad_after,
        )


def test_inference_requires_initialization(analysis):
    with pytest.raises(RuntimeError):
        analysis.fit()

    with pytest.raises(RuntimeError):
        analysis.profile([0.5, 0.6])


def test_initialize(analysis):
    result = analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    assert analysis.initialization is result

    assert result.start["eps_S"] == 0.7
    assert result.start["eps_B"] == 0.2

    assert set(result.start) == set(
        analysis.parameter_map.names
    )

    assert result.limits["eps_S"] == (0, 1)
    assert result.limits["eps_B"] == (0, 1)


def test_initialize_rejects_invalid_efficiency(analysis):
    with pytest.raises(ValueError):
        analysis.initialize(
            eps_S=1.2,
            eps_B=0.2,
        )

    with pytest.raises(ValueError):
        analysis.initialize(
            eps_S=0.8,
            eps_B=-0.1,
        )


def test_start_override(analysis):
    initial = analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    start = analysis._resolve_start(
        {"eps_S": 0.75}
    )

    assert start["eps_S"] == 0.75
    assert start["eps_B"] == 0.2
    assert start["A"] == initial.start["A"]


def test_unknown_start_parameter(analysis):
    analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    with pytest.raises(ValueError):
        analysis.fit(
            start={"banana": 1.0}
        )


def test_fit(analysis):
    analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    result = analysis.fit()

    assert result.valid
    assert np.isfinite(result.nll)

    assert set(result.values) == set(
        analysis.parameter_map.names
    )

    assert 0 <= result.values["eps_S"] <= 1
    assert 0 <= result.values["eps_B"] <= 1


def test_profile(analysis):
    analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    fit = analysis.fit()

    eps_hat = fit.values["eps_S"]

    poi_values = np.clip(
        np.array([
            eps_hat - 0.05,
            eps_hat,
            eps_hat + 0.05,
        ]),
        0,
        1,
    )

    results = analysis.profile(
        poi_values,
        start=fit.values,
    )

    assert len(results) == 3
    assert all(result.valid for result in results)

    np.testing.assert_allclose(
        [result.poi_value for result in results],
        poi_values,
    )

    assert all(
        result.global_fit is results[0].global_fit
        for result in results
    )


def test_builder(api_data):
    edges, _, _ = api_data

    config = GammaLineConfig(
        model="GaussPolynomial",
        degree=0,
    )

    builder = GammaLineProblemBuilder(config)
    problem = builder.build(edges)

    assert problem.parameter_map.names == (
        "eps_S",
        "eps_B",
        "A",
        "mu",
        "sig",
        "b0",
    )

def test_feldman_cousins_api(analysis, monkeypatch):
    analysis.initialize(
        eps_S=0.7,
        eps_B=0.2,
    )

    captured = {}

    class DummyFC:
        def __init__(
            self,
            problem,
            fitter,
            confidence_level,
            n_toys,
            seed,
            n_jobs,
        ):
            captured["problem"] = problem
            captured["confidence_level"] = confidence_level
            captured["n_toys"] = n_toys
            captured["seed"] = seed
            captured["n_jobs"] = n_jobs

        def run(self, data, poi_values, start=None, start_factory=None):
            captured["data"] = data
            captured["poi_values"] = np.asarray(poi_values)
            captured["start"] = start
            captured["start_factory"] = start_factory
            return "fc-result"

    monkeypatch.setattr(
        "pyinfer.api.FeldmanCousins",
        DummyFC,
    )

    result = analysis.feldman_cousins(
        [0.6, 0.7, 0.8],
        confidence_level=0.95,
        n_toys=123,
        seed=456,
        n_jobs=4,
    )

    assert result == "fc-result"
    assert captured["problem"] is analysis.problem
    assert captured["confidence_level"] == 0.95
    assert captured["n_toys"] == 123
    assert captured["seed"] == 456
    assert captured["n_jobs"] == 4

    np.testing.assert_array_equal(
        captured["data"],
        analysis.data,
    )

    np.testing.assert_array_equal(
        captured["poi_values"],
        [0.6, 0.7, 0.8],
    )
