from functools import partial

import numpy as np

from .builder import GammaLineProblemBuilder
from .config import GammaLineConfig
from .inference.feldman_cousins import FeldmanCousins
from .inference.fit import MinuitFitter
from .inference.profile import profile_scan


def _gamma_start_from_data(
    data,
    config,
    bin_edges,
    eps_S,
    eps_B,
    overrides,
):
    data = np.asarray(data)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must have shape (n_bins, 2)")

    builder = GammaLineProblemBuilder(config)

    initialization = builder.initialize(
        bin_edges,
        data[:, 0],
        eps_S,
        eps_B,
    )

    start = dict(initialization.start)
    start.update(overrides)

    return start


class GammaLineAnalysis:
    def __init__(self, bin_edges, before, after, config=None):
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.before = np.asarray(before, dtype=float)
        self.after = np.asarray(after, dtype=float)

        if self.bin_edges.ndim != 1:
            raise ValueError("bin_edges must be one-dimensional")
        if self.before.ndim != 1 or self.after.ndim != 1:
            raise ValueError("before and after must be one-dimensional")
        if len(self.before) != len(self.after):
            raise ValueError("before and after must have the same length")
        if len(self.bin_edges) != len(self.before) + 1:
            raise ValueError("bin_edges must have length len(before) + 1")

        if np.any(~np.isfinite(self.before)) or np.any(self.before < 0):
            raise ValueError("before counts must be finite and non-negative")
        if np.any(~np.isfinite(self.after)) or np.any(self.after < 0):
            raise ValueError("after counts must be finite and non-negative")
        if np.any(self.before != np.floor(self.before)):
            raise ValueError("before counts must be integers")
        if np.any(self.after != np.floor(self.after)):
            raise ValueError("after counts must be integers")
        if np.any(self.after > self.before):
            raise ValueError("after counts cannot exceed before counts")

        self.before = self.before.astype(int)
        self.after = self.after.astype(int)
        self.data = np.column_stack((self.before, self.after))

        self.config = GammaLineConfig() if config is None else config
        self.builder = GammaLineProblemBuilder(self.config)
        self.problem = self.builder.build(self.bin_edges)
        self._initialization = None

    @property
    def parameter_map(self):
        return self.problem.parameter_map

    @property
    def initialization(self):
        return self._initialization

    def initialize(self, eps_S, eps_B):
        self._initialization = self.builder.initialize(
            self.bin_edges,
            self.before,
            eps_S,
            eps_B,
        )
        return self._initialization

    def _require_initialization(self):
        if self._initialization is None:
            raise RuntimeError("Call initialize() before running inference")
        return self._initialization

    def _resolve_start(self, start):
        initialization = self._require_initialization()
        values = dict(initialization.start)

        if start is not None:
            unknown = [name for name in start if name not in values]
            if unknown:
                raise ValueError(f"Unknown initial parameters: {unknown}")
            values.update(start)

        return values

    def _resolve_start_overrides(self, start):
        initialization = self._require_initialization()

        if start is None:
            return {}

        unknown = [name for name in start if name not in initialization.start]
        if unknown:
            raise ValueError(f"Unknown initial parameters: {unknown}")

        return dict(start)

    def _make_start_factory(self, start=None):
        initialization = self._require_initialization()

        return partial(
            _gamma_start_from_data,
            config=self.config,
            bin_edges=self.bin_edges,
            eps_S=initialization.start["eps_S"],
            eps_B=initialization.start["eps_B"],
            overrides=self._resolve_start_overrides(start),
        )

    def _resolve_limits(self, limits):
        initialization = self._require_initialization()
        values = dict(initialization.limits)

        if limits is not None:
            unknown = [name for name in limits if name not in values]
            if unknown:
                raise ValueError(f"Unknown parameter limits: {unknown}")
            values.update(limits)

        return values

    def _make_fitter(self, limits=None):
        return MinuitFitter(
            self.problem,
            limits=self._resolve_limits(limits),
        )

    def fit(
        self,
        start=None,
        limits=None,
        hesse=False,
        retry=True,
        ncall=None,
        strategy=1,
        tol=None,
        iterate=5,
        use_simplex=True,
        retry_strategy=2,
    ):
        fitter = self._make_fitter(limits)

        return fitter.fit(
            self.data,
            start=self._resolve_start(start),
            hesse=hesse,
            retry=retry,
            ncall=ncall,
            strategy=strategy,
            tol=tol,
            iterate=iterate,
            use_simplex=use_simplex,
            retry_strategy=retry_strategy,
        )

    def profile(
        self,
        poi_values,
        start=None,
        limits=None,
        fit_options=None,
    ):
        fitter = self._make_fitter(limits)

        return profile_scan(
            fitter,
            self.data,
            poi_values,
            start=self._resolve_start(start),
            fit_options=fit_options,
        )

    def feldman_cousins(
        self,
        poi_values,
        confidence_level=0.9,
        n_toys=1000,
        seed=None,
        n_jobs=1,
        start=None,
        limits=None,
        fit_options=None,
    ):
        fitter = self._make_fitter(limits)

        fc_kwargs = {
            "confidence_level": confidence_level,
            "n_toys": n_toys,
            "seed": seed,
            "n_jobs": n_jobs,
        }

        if fit_options is not None:
            fc_kwargs["fit_options"] = fit_options

        fc = FeldmanCousins(
            self.problem,
            fitter,
            **fc_kwargs,
        )

        return fc.run(
            self.data,
            poi_values,
            start_factory=self._make_start_factory(start),
        )
