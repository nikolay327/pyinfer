from dataclasses import dataclass
import numpy as np

from .config import GammaLineConfig
from .inference.initialize import get_initial_params
from .inference.parameters import get_parameter_map
from .inference.problem import InferenceProblem
from .likelihood.binned_gamma import BinnedLikelihood
from .nuisance_model.sig_plus_bg import (
    GaussEMGLeftPolyStep,
    GaussEMGLeftPolynomial,
    GaussPolyStep,
    GaussPolynomial,
)


MODEL_CLASSES = {
    "GaussEMGLeftPolyStep": GaussEMGLeftPolyStep,
    "GaussEMGLeftPolynomial": GaussEMGLeftPolynomial,
    "GaussPolyStep": GaussPolyStep,
    "GaussPolynomial": GaussPolynomial,
}


@dataclass
class GammaLineInitialization:
    start: dict
    limits: dict
    optimizer_result: object


class GammaLineProblemBuilder:
    def __init__(self, config: GammaLineConfig):
        if not isinstance(config, GammaLineConfig):
            raise TypeError("config must be a GammaLineConfig")
        self.config = config
        self.parameter_map = get_parameter_map(config.model, degree=config.degree)

    def build(self, bin_edges):
        edges = np.asarray(bin_edges, dtype=float)
        x_ref = 0.5 * (edges[0] + edges[-1])
        x_scale = 0.5 * (edges[-1] - edges[0])
        model = MODEL_CLASSES[self.config.model](x_ref=x_ref, x_scale=x_scale)
        likelihood = BinnedLikelihood(model.signal, model.background, edges)
        return InferenceProblem(likelihood, self.parameter_map)

    def initialize(self, bin_edges, before, eps_S, eps_B):
        if not np.isfinite(eps_S) or not 0 <= eps_S <= 1:
            raise ValueError("eps_S must be finite and between 0 and 1")
        if not np.isfinite(eps_B) or not 0 <= eps_B <= 1:
            raise ValueError("eps_B must be finite and between 0 and 1")

        params, limits, result = get_initial_params(
            self.config.model, bin_edges, before, degree=self.config.degree
        )
        start = self.parameter_map.complete(params, eps_S, eps_B)
        return GammaLineInitialization(start=start, limits=limits, optimizer_result=result)
