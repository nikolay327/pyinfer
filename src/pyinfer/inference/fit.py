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
    edm_goal: float
    nfcn: int
    algorithm: str
    reached_call_limit: bool
    above_max_edm: bool
    has_covariance: bool
    accurate_covar: bool
    posdef_covar: bool
    made_posdef_covar: bool
    parameters_at_limit: bool
    hesse_failed: bool

    @property
    def failure_reason(self):
        reasons = []
        if self.reached_call_limit:
            reasons.append("call limit")
        if self.above_max_edm:
            reasons.append("EDM above threshold")
        if not np.isfinite(self.nll):
            reasons.append("non-finite NLL")
        if not self.valid and not reasons:
            reasons.append("invalid minimum")
        return ", ".join(reasons) if reasons else None


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

    def fit(
        self,
        data,
        start,
        fixed=None,
        hesse=False,
        retry=True,
        ncall=None,
        strategy=1,
        tol=None,
        iterate=5,
        use_simplex=True,
        retry_strategy=2,
    ):
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

        if strategy not in (0, 1, 2):
            raise ValueError("strategy must be 0, 1, or 2")
        if retry_strategy not in (0, 1, 2):
            raise ValueError("retry_strategy must be 0, 1, or 2")
        if not isinstance(iterate, int) or iterate <= 0:
            raise ValueError("iterate must be a positive integer")
        if tol is not None and (not np.isfinite(tol) or tol <= 0):
            raise ValueError("tol must be finite and positive")

        for name, value in fixed.items():
            start[name] = value

        if any(not np.isfinite(start[name]) for name in names):
            raise ValueError("Initial parameters must be finite")

        minuit = Minuit(
            self._objective(data),
            *(start[name] for name in names),
            name=names,
        )

        minuit.errordef = Minuit.LIKELIHOOD
        minuit.strategy = strategy

        if tol is not None:
            minuit.tol = tol

        for name, limit in self.limits.items():
            minuit.limits[name] = limit

        for name in fixed:
            minuit.fixed[name] = True

        minuit.migrad(
            ncall=ncall,
            iterate=iterate,
            use_simplex=use_simplex,
        )

        if retry and not minuit.valid:
            minuit.strategy = retry_strategy
            minuit.simplex(ncall=ncall)
            minuit.migrad(
                ncall=ncall,
                iterate=iterate,
                use_simplex=use_simplex,
            )

        if hesse and minuit.valid:
            minuit.hesse()

        fmin = minuit.fmin

        return FitResult(
            values={name: float(minuit.values[name]) for name in names},
            errors={name: float(minuit.errors[name]) for name in names},
            nll=float(minuit.fval),
            valid=bool(minuit.valid and np.isfinite(minuit.fval)),
            edm=float(fmin.edm),
            edm_goal=float(fmin.edm_goal),
            nfcn=minuit.nfcn,
            algorithm=fmin.algorithm,
            reached_call_limit=fmin.has_reached_call_limit,
            above_max_edm=fmin.is_above_max_edm,
            has_covariance=fmin.has_covariance,
            accurate_covar=fmin.has_accurate_covar,
            posdef_covar=fmin.has_posdef_covar,
            made_posdef_covar=fmin.has_made_posdef_covar,
            parameters_at_limit=fmin.has_parameters_at_limit,
            hesse_failed=fmin.hesse_failed,
        )
