from dataclasses import dataclass

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
        return self.global_fit.valid and self.conditional_fit.valid


def profile_likelihood_ratio(fitter: MinuitFitter, data, poi_value: float, start=None, global_fit=None):
    poi = fitter.parameter_map.poi

    if global_fit is None:
        if start is None:
            raise ValueError("start must be provided when global_fit is not given")

        global_fit = fitter.fit(data, start)

    conditional_fit = fitter.fit(
        data,
        start=global_fit.values,
        fixed={poi: poi_value},
    )

    q = max(0.0, 2.0 * (conditional_fit.nll - global_fit.nll))

    return ProfileResult(
        poi_value=poi_value,
        q=q,
        global_fit=global_fit,
        conditional_fit=conditional_fit,
    )


def profile_scan(fitter: MinuitFitter, data, poi_values, start):
    global_fit = fitter.fit(data, start)
    conditional_start = global_fit.values
    poi = fitter.parameter_map.poi

    results = []

    for poi_value in poi_values:
        conditional_fit = fitter.fit(
            data,
            start=conditional_start,
            fixed={poi: poi_value},
        )

        q = max(0.0, 2.0 * (conditional_fit.nll - global_fit.nll))

        result = ProfileResult(
            poi_value=poi_value,
            q=q,
            global_fit=global_fit,
            conditional_fit=conditional_fit,
        )

        results.append(result)

        if conditional_fit.valid:
            conditional_start = conditional_fit.values

    return results
