import numpy as np
from dataclasses import dataclass
from iminuit import Minuit


@dataclass
class FitResult:
    values: dict
    errors: dict
    nll: float
    valid: bool
    edm: float
    nfcn: int


class MinuitFitter:
    def __init__(self, problem, limits=None):
        self.problem = problem
        self.parameter_map = problem.parameter_map
        self.limits = {} if limits is None else dict(limits)

    def _objective(self, data):
        def nll(*values):
            pars = self.parameter_map.from_vector(values)

            try:
                value = self.problem.nll(data, pars)
            except ValueError:
                return np.inf

            if not np.isfinite(value):
                return np.inf

            return float(value)

        return nll

    def fit(self, data, start, fixed=None, hesse=False, retry=True):
        start = dict(start)
        fixed = {} if fixed is None else dict(fixed)

        names = self.parameter_map.names

        missing = [name for name in names if name not in start]
        if missing:
            raise ValueError(f"Missing initial parameters: {missing}")

        unknown_fixed = [name for name in fixed if name not in names]
        if unknown_fixed:
            raise ValueError(f"Unknown fixed parameters: {unknown_fixed}")

        unknown_limits = [name for name in self.limits if name not in names]
        if unknown_limits:
            raise ValueError(f"Unknown parameter limits: {unknown_limits}")

        for name, value in fixed.items():
            start[name] = value

        values = [start[name] for name in names]

        minuit = Minuit(self._objective(data), *values, name=names)
        minuit.errordef = Minuit.LIKELIHOOD

        for name, limit in self.limits.items():
            minuit.limits[name] = limit

        for name in fixed:
            minuit.fixed[name] = True

        minuit.migrad()

        if retry and not minuit.valid:
            minuit.simplex()
            minuit.migrad()

        if hesse and minuit.valid:
            minuit.hesse()

        return FitResult(
            values={name: minuit.values[name] for name in names},
            errors={name: minuit.errors[name] for name in names},
            nll=float(minuit.fval),
            valid=minuit.valid,
            edm=float(minuit.fmin.edm),
            nfcn=minuit.nfcn,
        )
