from abc import ABC, abstractmethod

class LikelihoodBase(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def sample(self, size=1):
        pass

class Sequential(LikelihoodBase):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.likelihoods = list(args)

    def append(self, likelihood: LikelihoodBase):
        self.likelihoods.append(likelihood)

    def __call__(self, *args, reduction="sum"):
        assert len(args) == len(self.likelihoods), (
            "Number of arguments must match number of likelihoods"
        )
        assert len(self.likelihoods) > 0, "No likelihoods to evaluate"

        log_likelihoods = [
            likelihood(*arg) if isinstance(arg, tuple) else likelihood(arg)
            for likelihood, arg in zip(self.likelihoods, args)
        ]

        if reduction == "sum":
            return sum(log_likelihoods)
        elif reduction == "mean":
            return sum(log_likelihoods) / len(log_likelihoods)
        else:
            return log_likelihoods

    def sample(self, size=1):
        assert len(self.likelihoods) > 0, "No likelihoods to sample from"

        samples = [likelihood.sample(size=size) for likelihood in self.likelihoods]
        return samples
