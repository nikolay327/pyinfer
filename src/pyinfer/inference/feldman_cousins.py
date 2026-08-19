from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

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
    seed_entropy: object
    points: list

    @property
    def poi_values(self):
        return np.asarray([point.poi_value for point in self.points])

    @property
    def accepted_values(self):
        return np.asarray([
            point.poi_value
            for point in self.points
            if point.accepted
        ])

    @property
    def p_values(self):
        return np.asarray([point.p_value for point in self.points])

    @property
    def q_obs(self):
        return np.asarray([point.q_obs for point in self.points])

    @property
    def q_crit(self):
        return np.asarray([point.q_crit for point in self.points])


def _get_start(data, start, start_factory):
    if start_factory is not None:
        return dict(start_factory(data))
    return dict(start)


def _run_fc_point(
    problem,
    fitter,
    profile,
    start,
    start_factory,
    seed_sequence,
    n_toys,
    confidence_level,
    fit_options,
):
    generation_params = dict(profile.conditional_fit.values)

    toy_seeds = seed_sequence.spawn(n_toys)
    q_toys = np.full(n_toys, np.nan)

    for i, toy_seed in enumerate(toy_seeds):
        rng = np.random.default_rng(toy_seed)

        toy = problem.sample(
            generation_params,
            size=1,
            rng=rng,
        )[0]

        toy_start = _get_start(
            toy,
            start,
            start_factory,
        )

        result = profile_likelihood_ratio(
            fitter,
            toy,
            profile.poi_value,
            start=toy_start,
            fit_options=fit_options,
        )

        if np.isfinite(result.q):
            q_toys[i] = result.q

    failed = np.flatnonzero(~np.isfinite(q_toys))

    if len(failed):
        raise RuntimeError(
            f"{len(failed)}/{n_toys} toys returned non-finite test statistics "
            f"for {fitter.parameter_map.poi}={profile.poi_value}. "
            f"Toy indices: {failed.tolist()}"
        )

    q_crit = np.quantile(
        q_toys,
        confidence_level,
        method="higher",
    )

    p_value = (
        np.count_nonzero(q_toys >= profile.q) + 1
    ) / (n_toys + 1)

    return FCPointResult(
        poi_value=profile.poi_value,
        q_obs=float(profile.q),
        q_crit=float(q_crit),
        p_value=float(p_value),
        accepted=bool(profile.q <= q_crit),
        generation_params=generation_params,
        q_toys=q_toys,
        n_toys=n_toys,
        n_valid=n_toys,
        n_failed=0,
    )


class FeldmanCousins:
    def __init__(
        self,
        problem,
        fitter,
        confidence_level=0.9,
        n_toys=1000,
        seed=None,
        n_jobs=1,
        fit_options=None,
    ):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if not isinstance(n_toys, int) or n_toys <= 0:
            raise ValueError("n_toys must be a positive integer")

        if n_jobs is not None and (
            not isinstance(n_jobs, int) or n_jobs <= 0
        ):
            raise ValueError("n_jobs must be a positive integer or None")

        self.problem = problem
        self.fitter = fitter
        self.confidence_level = confidence_level
        self.n_toys = n_toys
        self.seed = seed
        self.n_jobs = n_jobs
        self.fit_options = {} if fit_options is None else dict(fit_options)

    def _run_serial(self, profiles, start, start_factory, point_seeds):
        return [
            _run_fc_point(
                self.problem,
                self.fitter,
                profile,
                start,
                start_factory,
                point_seed,
                self.n_toys,
                self.confidence_level,
                self.fit_options,
            )
            for profile, point_seed in zip(profiles, point_seeds)
        ]

    def _run_parallel(self, profiles, start, start_factory, point_seeds):
        points = [None] * len(profiles)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _run_fc_point,
                    self.problem,
                    self.fitter,
                    profile,
                    start,
                    start_factory,
                    point_seed,
                    self.n_toys,
                    self.confidence_level,
                    self.fit_options,
                ): i
                for i, (profile, point_seed) in enumerate(
                    zip(profiles, point_seeds)
                )
            }

            for future in as_completed(futures):
                i = futures[future]
                points[i] = future.result()

        return points

    def run(
        self,
        data,
        poi_values,
        start=None,
        start_factory=None,
    ):
        poi_values = np.asarray(poi_values, dtype=float)

        if poi_values.ndim != 1 or len(poi_values) == 0:
            raise ValueError("poi_values must be a non-empty one-dimensional array")

        if np.any(~np.isfinite(poi_values)):
            raise ValueError("poi_values must be finite")

        if start is None and start_factory is None:
            raise ValueError("Either start or start_factory must be provided")

        if start is not None and start_factory is not None:
            raise ValueError("Provide either start or start_factory, not both")

        observed_start = _get_start(
            data,
            start,
            start_factory,
        )

        profiles = profile_scan(
            self.fitter,
            data,
            poi_values,
            start=observed_start,
            fit_options=self.fit_options,
        )

        invalid = [
            profile.poi_value
            for profile in profiles
            if not profile.valid
        ]

        if invalid:
            raise RuntimeError(
                f"Observed profile returned non-finite values for "
                f"{self.fitter.parameter_map.poi}={invalid}"
            )

        root_seed = np.random.SeedSequence(self.seed)
        point_seeds = root_seed.spawn(len(profiles))

        if self.n_jobs == 1:
            points = self._run_serial(
                profiles,
                start,
                start_factory,
                point_seeds,
            )
        else:
            points = self._run_parallel(
                profiles,
                start,
                start_factory,
                point_seeds,
            )

        return FeldmanCousinsResult(
            confidence_level=self.confidence_level,
            seed_entropy=root_seed.entropy,
            points=points,
        )
