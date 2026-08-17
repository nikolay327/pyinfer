import warnings
import numpy as np
from dataclasses import dataclass

from .profile import profile_scan


@dataclass
class FCPointResult:
    poi_value: float
    q_obs: float
    q_crit: float
    p_value: float
    accepted: bool
    generation_params: dict
    q_toys: np.ndarray
    n_toys: int
    n_valid: int
    n_failed: int


@dataclass
class FeldmanCousinsResult:
    confidence_level: float
    points: list

    @property
    def accepted_values(self):
        return np.asarray([point.poi_value for point in self.points if point.accepted])

    @property
    def p_values(self):
        return np.asarray([point.p_value for point in self.points])

    @property
    def q_obs(self):
        return np.asarray([point.q_obs for point in self.points])

    @property
    def q_crit(self):
        return np.asarray([point.q_crit for point in self.points])


class FeldmanCousins:
    def __init__(self, problem, fitter, confidence_level=0.9, n_toys=1000, seed=None):
        self.problem = problem
        self.fitter = fitter
        self.confidence_level = confidence_level
        self.n_toys = n_toys
        self.seed = seed

    @staticmethod
    def _format_toys(samples):
        if isinstance(samples, np.ndarray):
            return samples

        toys = [
            np.stack((before, after), axis=-1)
            for before, after in samples
        ]

        return np.stack(toys, axis=1)

    def _toy_statistics(self, poi_value, generation_params):
        samples = self.problem.sample(generation_params, size=self.n_toys)
        toys = self._format_toys(samples)

        q_toys = []
        n_failed = 0
        poi = self.fitter.parameter_map.poi

        for toy in toys:
            global_fit = self.fitter.fit(toy, start=generation_params)

            if not global_fit.valid:
                n_failed += 1
                continue

            conditional_fit = self.fitter.fit(
                toy,
                start=global_fit.values,
                fixed={poi: poi_value},
            )

            if not conditional_fit.valid:
                n_failed += 1
                continue

            q = max(0.0, 2.0 * (conditional_fit.nll - global_fit.nll))
            q_toys.append(q)

        return np.asarray(q_toys), n_failed

    def run(self, data, poi_values, start):
        if self.seed is not None:
            np.random.seed(self.seed)

        profiles = profile_scan(
            self.fitter,
            data,
            poi_values,
            start,
        )

        points = []

        for profile in profiles:
            if not profile.valid:
                raise RuntimeError(f"Observed fit failed for {self.fitter.parameter_map.poi}={profile.poi_value}")

            generation_params = dict(profile.conditional_fit.values)

            q_toys, n_failed = self._toy_statistics(
                profile.poi_value,
                generation_params,
            )

            if len(q_toys) == 0:
                raise RuntimeError(f"All toy fits failed for {self.fitter.parameter_map.poi}={profile.poi_value}")

            if n_failed > 0:
                warnings.warn(
                    f"{n_failed}/{self.n_toys} toy fits failed for "
                    f"{self.fitter.parameter_map.poi}={profile.poi_value}"
                )

            q_crit = np.quantile(q_toys, self.confidence_level, method="higher")
            p_value = (np.count_nonzero(q_toys >= profile.q) + 1) / (len(q_toys) + 1)

            points.append(
                FCPointResult(
                    poi_value=profile.poi_value,
                    q_obs=profile.q,
                    q_crit=q_crit,
                    p_value=p_value,
                    accepted=profile.q <= q_crit,
                    generation_params=generation_params,
                    q_toys=q_toys,
                    n_toys=self.n_toys,
                    n_valid=len(q_toys),
                    n_failed=n_failed,
                )
            )

        return FeldmanCousinsResult(
            confidence_level=self.confidence_level,
            points=points,
        )
