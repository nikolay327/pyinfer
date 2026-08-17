from abc import ABC, abstractmethod


class LikelihoodBase(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def sample(self, size=1, rng=None):
        pass


class Sequential(LikelihoodBase):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.likelihoods = list(args)

    def append(self, likelihood: LikelihoodBase):
        self.likelihoods.append(likelihood)

    def __call__(self, *args, reduction="sum"):
        if len(args) != len(self.likelihoods):
            raise ValueError("Number of arguments must match number of likelihoods")
        if not self.likelihoods:
            raise ValueError("No likelihoods to evaluate")

        values = [
            likelihood(*arg) if isinstance(arg, tuple) else likelihood(arg)
            for likelihood, arg in zip(self.likelihoods, args)
        ]

        if reduction == "sum":
            return sum(values)
        if reduction == "mean":
            return sum(values) / len(values)
        if reduction is None:
            return values
        raise ValueError(f"Unknown reduction: {reduction}")

    def sample(self, size=1, rng=None):
        if not self.likelihoods:
            raise ValueError("No likelihoods to sample from")
        return [likelihood.sample(size=size, rng=rng) for likelihood in self.likelihoods]
