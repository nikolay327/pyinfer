from dataclasses import dataclass
import numpy as np

from .fit import FitResult, MinuitFitter


@dataclass
class ProfileResult:
    poi_value: float
    q: float
    global_fit: FitResult
    conditional_fit: FitResult

    @property
    def delta_nll(self):
        return self.conditional_fit.nll - self.global_fit.nll

    @property
    def valid(self):
        return (
            self.global_fit.valid
            and self.conditional_fit.valid
            and np.isfinite(self.q)
        )


def _compute_q(global_fit, conditional_fit, tol=1e-7):
    if not global_fit.valid or not conditional_fit.valid:
        return np.nan

    delta = conditional_fit.nll - global_fit.nll
    scale = max(1.0, abs(global_fit.nll), abs(conditional_fit.nll))

    if delta < -tol * scale:
        return np.nan

    return max(0.0, 2.0 * delta)


def profile_likelihood_ratio(
    fitter: MinuitFitter,
    data,
    poi_value: float,
    start=None,
    global_fit=None,
    tol=1e-7,
):
    poi = fitter.parameter_map.poi

    if global_fit is None:
        if start is None:
            raise ValueError("start must be provided when global_fit is not given")
        global_fit = fitter.fit(data, start)

    if not global_fit.valid:
        return ProfileResult(
            poi_value=poi_value,
            q=np.nan,
            global_fit=global_fit,
            conditional_fit=global_fit,
        )

    conditional_fit = fitter.fit(
        data,
        start=global_fit.values,
        fixed={poi: poi_value},
    )

    q = _compute_q(global_fit, conditional_fit, tol)

    return ProfileResult(
        poi_value=poi_value,
        q=q,
        global_fit=global_fit,
        conditional_fit=conditional_fit,
    )


def profile_scan(fitter: MinuitFitter, data, poi_values, start, tol=1e-7):
    global_fit = fitter.fit(data, start)

    if not global_fit.valid:
        raise RuntimeError(
            f"Global fit failed: {global_fit.failure_reason}"
        )

    poi = fitter.parameter_map.poi
    conditional_start = global_fit.values
    results = []

    for poi_value in poi_values:
        conditional_fit = fitter.fit(
            data,
            start=conditional_start,
            fixed={poi: poi_value},
        )

        if not conditional_fit.valid:
            conditional_fit = fitter.fit(
                data,
                start=global_fit.values,
                fixed={poi: poi_value},
            )

        q = _compute_q(global_fit, conditional_fit, tol)

        results.append(
            ProfileResult(
                poi_value=poi_value,
                q=q,
                global_fit=global_fit,
                conditional_fit=conditional_fit,
            )
        )

        if conditional_fit.valid and np.isfinite(q):
            conditional_start = conditional_fit.values

    return results
