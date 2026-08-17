class InferenceProblem:
    def __init__(self, likelihood, parameter_map):
        self.likelihood = likelihood
        self.parameter_map = parameter_map

    def log_likelihood(self, data, pars):
        eps_S, eps_B, sig_pars, bg_pars = self.parameter_map.split(pars)
        return self.likelihood(data, eps_S, eps_B, sig_pars, bg_pars)

    def nll(self, data, pars):
        return -self.log_likelihood(data, pars)

    def sample(self, pars, size=1, rng=None):
        eps_S, eps_B, sig_pars, bg_pars = self.parameter_map.split(pars)
        return self.likelihood.sample(eps_S, eps_B, sig_pars, bg_pars, size=size, rng=rng)
