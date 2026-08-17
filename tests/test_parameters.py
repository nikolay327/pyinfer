import pytest

from pyinfer.inference.parameters import get_parameter_map


def test_parameter_names():
    parameter_map = get_parameter_map(
        "GaussEMGLeftPolyStep",
        degree=1,
    )

    assert parameter_map.names == (
        "eps_S",
        "eps_B",
        "A",
        "mu",
        "sig",
        "f_tail",
        "tau",
        "h_step",
        "c0",
        "c1",
    )

    assert parameter_map.shared_names == ("mu", "sig")


def test_split():
    parameter_map = get_parameter_map(
        "GaussEMGLeftPolyStep",
        degree=1,
    )

    pars = {
        "eps_S": 0.8,
        "eps_B": 0.2,
        "A": 100.0,
        "mu": 1.0,
        "sig": 0.5,
        "f_tail": 0.1,
        "tau": 0.8,
        "h_step": 0.2,
        "c0": 3.0,
        "c1": 0.1,
    }

    eps_S, eps_B, sig_pars, bg_pars = parameter_map.split(pars)

    assert eps_S == 0.8
    assert eps_B == 0.2
    assert sig_pars == (100.0, 1.0, 0.5, 0.1, 0.8)
    assert bg_pars == (1.0, 0.5, 0.2, 3.0, 0.1)


def test_merge_shared_parameters():
    parameter_map = get_parameter_map(
        "GaussPolyStep",
        degree=0,
    )

    pars = parameter_map.merge(
        0.8,
        0.2,
        (100.0, 1.0, 0.5),
        (1.0, 0.5, 0.2, 3.0),
    )

    assert pars["mu"] == 1.0
    assert pars["sig"] == 0.5


def test_merge_rejects_inconsistent_shared_parameter():
    parameter_map = get_parameter_map(
        "GaussPolyStep",
        degree=0,
    )

    with pytest.raises(ValueError):
        parameter_map.merge(
            0.8,
            0.2,
            (100.0, 1.0, 0.5),
            (1.1, 0.5, 0.2, 3.0),
        )


def test_vector_roundtrip():
    parameter_map = get_parameter_map(
        "GaussPolynomial",
        degree=0,
    )

    pars = {
        "eps_S": 0.8,
        "eps_B": 0.2,
        "A": 100.0,
        "mu": 1.0,
        "sig": 0.5,
        "c0": 3.0,
    }

    assert parameter_map.from_vector(
        parameter_map.to_vector(pars)
    ) == pars


def test_invalid_degree():
    with pytest.raises(ValueError):
        get_parameter_map("GaussPolynomial", degree=-1)
