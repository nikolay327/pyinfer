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
            np.isfinite(self.global_fit.nll)
            and np.isfinite(self.conditional_fit.nll)
            and np.isfinite(self.q)
        )


def _compute_q(global_fit, conditional_fit):
    if not np.isfinite(global_fit.nll) or not np.isfinite(conditional_fit.nll):
        return np.nan

    return 2.0 * (conditional_fit.nll - global_fit.nll)


def profile_likelihood_ratio(
    fitter: MinuitFitter,
    data,
    poi_value: float,
    start,
    global_fit=None,
    fit_options=None,
):
    poi = fitter.parameter_map.poi
    fit_options = {} if fit_options is None else dict(fit_options)

    if global_fit is None:
        global_fit = fitter.fit(
            data,
            start=start,
            **fit_options,
        )

    conditional_fit = fitter.fit(
        data,
        start=start,
        fixed={poi: poi_value},
        **fit_options,
    )

    q = _compute_q(global_fit, conditional_fit)

    return ProfileResult(
        poi_value=poi_value,
        q=q,
        global_fit=global_fit,
        conditional_fit=conditional_fit,
    )


def profile_scan(
    fitter: MinuitFitter,
    data,
    poi_values,
    start,
    fit_options=None,
):
    poi_values = np.asarray(poi_values, dtype=float)
    fit_options = {} if fit_options is None else dict(fit_options)

    if poi_values.ndim != 1 or len(poi_values) == 0:
        raise ValueError("poi_values must be a non-empty one-dimensional array")

    if np.any(~np.isfinite(poi_values)):
        raise ValueError("poi_values must be finite")

    global_fit = fitter.fit(
        data,
        start=start,
        **fit_options,
    )

    if not np.isfinite(global_fit.nll):
        raise RuntimeError("Global fit returned a non-finite NLL")

    poi = fitter.parameter_map.poi
    results = []

    for poi_value in poi_values:
        conditional_fit = fitter.fit(
            data,
            start=start,
            fixed={poi: poi_value},
            **fit_options,
        )

        q = _compute_q(global_fit, conditional_fit)

        results.append(
            ProfileResult(
                poi_value=poi_value,
                q=q,
                global_fit=global_fit,
                conditional_fit=conditional_fit,
            )
        )

    return results
