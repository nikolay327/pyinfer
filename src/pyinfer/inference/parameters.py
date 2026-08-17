import numpy as np


class ParameterMap:
    def __init__(self, signal_names, background_names, poi="eps_S"):
        self.signal_names = tuple(signal_names)
        self.background_names = tuple(background_names)
        self.poi = poi

        names = ("eps_S", "eps_B") + self.signal_names + self.background_names
        self.names = tuple(dict.fromkeys(names))

    @property
    def nuisance_names(self):
        return tuple(name for name in self.names if name != self.poi)

    @property
    def shared_names(self):
        return tuple(name for name in self.signal_names if name in self.background_names)

    def split(self, pars):
        eps_S = pars["eps_S"]
        eps_B = pars["eps_B"]
        sig_pars = tuple(pars[name] for name in self.signal_names)
        bg_pars = tuple(pars[name] for name in self.background_names)
        return eps_S, eps_B, sig_pars, bg_pars

    def merge(self, eps_S, eps_B, sig_pars, bg_pars):
        if len(sig_pars) != len(self.signal_names):
            raise ValueError("Incorrect number of signal parameters")

        if len(bg_pars) != len(self.background_names):
            raise ValueError("Incorrect number of background parameters")

        pars = {"eps_S": eps_S, "eps_B": eps_B}

        for name, value in zip(self.signal_names, sig_pars):
            pars[name] = value

        for name, value in zip(self.background_names, bg_pars):
            if name in pars and not np.isclose(pars[name], value):
                raise ValueError(f"Inconsistent shared parameter: {name}")
            pars[name] = value

        return {name: pars[name] for name in self.names}

    def complete(self, initial_params, eps_S, eps_B):
        pars = dict(initial_params)
        pars["eps_S"] = eps_S
        pars["eps_B"] = eps_B

        missing = [name for name in self.names if name not in pars]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        return {name: pars[name] for name in self.names}

    def to_vector(self, pars):
        return np.asarray([pars[name] for name in self.names], dtype=float)

    def from_vector(self, values):
        if len(values) != len(self.names):
            raise ValueError("Incorrect number of parameters")
        return dict(zip(self.names, values))


def get_parameter_map(model_name, degree=1):
    coeffs = tuple(f"c{i}" for i in range(degree + 1))

    if model_name == "GaussEMGLeftPolyStep":
        signal_names = ("A", "mu", "sig", "f_tail", "tau")
        background_names = ("mu", "sig", "h_step", *coeffs)

    elif model_name == "GaussEMGLeftPolynomial":
        signal_names = ("A", "mu", "sig", "f_tail", "tau")
        background_names = coeffs

    elif model_name == "GaussPolyStep":
        signal_names = ("A", "mu", "sig")
        background_names = ("mu", "sig", "h_step", *coeffs)

    elif model_name == "GaussPolynomial":
        signal_names = ("A", "mu", "sig")
        background_names = coeffs

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ParameterMap(signal_names, background_names)
