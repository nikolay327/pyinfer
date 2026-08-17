from dataclasses import dataclass

SUPPORTED_MODELS = (
    "GaussEMGLeftPolyStep",
    "GaussEMGLeftPolynomial",
    "GaussPolyStep",
    "GaussPolynomial",
)


@dataclass(frozen=True)
class GammaLineConfig:
    model: str = "GaussPolynomial"
    degree: int = 1

    def __post_init__(self):
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a non-negative integer")
