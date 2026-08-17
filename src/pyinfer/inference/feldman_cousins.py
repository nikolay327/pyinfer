import numpy as np
from dataclasses import dataclass

from .profile import profile_likelihood_ratio, profile_scan


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
    def poi_values(self):
        return np.asarray([point.poi_value for point in self.points])

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
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if not isinstance(n_toys, int) or n_toys <= 0:
            raise ValueError("n_toys must be a positive integer")

        self.problem = problem
        self.fitter = fitter
        self.confidence_level = confidence_level
        self.n_toys = n_toys
        self.seed = seed

    def _toy_statistic(self, toy, poi_value, generation_params):
        result = profile_likelihood_ratio(
            self.fitter,
            toy,
            poi_value,
            start=generation_params,
        )

        if not result.valid:
            return np.nan

        return result.q

    def _toy_statistics(self, poi_value, generation_params, seed_sequence):
        toy_seeds = seed_sequence.spawn(self.n_toys)
        q_toys = np.full(self.n_toys, np.nan)

        for i, seed in enumerate(toy_seeds):
            rng = np.random.default_rng(seed)

            toy = self.problem.sample(
                generation_params,
                size=1,
                rng=rng,
            )[0]

            q_toys[i] = self._toy_statistic(
                toy,
                poi_value,
                generation_params,
            )

        return q_toys

    def run(self, data, poi_values, start):
        poi_values = np.asarray(poi_values, dtype=float)

        if poi_values.ndim != 1 or len(poi_values) == 0:
            raise ValueError("poi_values must be a non-empty one-dimensional array")

        if np.any(~np.isfinite(poi_values)):
            raise ValueError("poi_values must be finite")

        profiles = profile_scan(
            self.fitter,
            data,
            poi_values,
            start,
        )

        root_seed = np.random.SeedSequence(self.seed)
        point_seeds = root_seed.spawn(len(profiles))

        points = []

        for profile, point_seed in zip(profiles, point_seeds):
            if not profile.valid:
                raise RuntimeError(
                    f"Observed profile fit failed for "
                    f"{self.fitter.parameter_map.poi}={profile.poi_value}"
                )

            generation_params = dict(profile.conditional_fit.values)

            q_toys = self._toy_statistics(
                profile.poi_value,
                generation_params,
                point_seed,
            )

            failed = np.flatnonzero(~np.isfinite(q_toys))

            if len(failed) > 0:
                raise RuntimeError(
                    f"{len(failed)}/{self.n_toys} toy fits failed for "
                    f"{self.fitter.parameter_map.poi}={profile.poi_value}. "
                    f"Failed toy indices: {failed.tolist()}"
                )

            q_crit = np.quantile(
                q_toys,
                self.confidence_level,
                method="higher",
            )

            p_value = (
                np.count_nonzero(q_toys >= profile.q) + 1
            ) / (self.n_toys + 1)

            points.append(
                FCPointResult(
                    poi_value=profile.poi_value,
                    q_obs=profile.q,
                    q_crit=float(q_crit),
                    p_value=float(p_value),
                    accepted=profile.q <= q_crit,
                    generation_params=generation_params,
                    q_toys=q_toys,
                    n_toys=self.n_toys,
                    n_valid=self.n_toys,
                    n_failed=0,
                )
            )

        return FeldmanCousinsResult(
            confidence_level=self.confidence_level,
            points=points,
        )
