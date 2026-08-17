from abc import ABC, abstractmethod

class ModelBase(ABC):
    @abstractmethod
    def __call__(self, x, *pars):
        pass

    @abstractmethod
    def integral(self, x_lo, x_hi, *pars):
        pass
