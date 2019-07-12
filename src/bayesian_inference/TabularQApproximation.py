import numpy as np
from scipy.stats import t
from src.bayesian_inference.bayesianapproximator import TabularBayesianApproximation

class TabularQApproximation(TabularBayesianApproximation):
    def __init__(self, state_dimensions, num_acts, gamma, params={'mu_0': 1.0, 'nu_0': 1.0, 'alpha_0': 1.5, 'beta_0': 2.0}):
        super().__init__(state_dimensions, num_acts, params)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))
        self.n = np.zeros((num_states * num_acts))

        self.mu_0 = params['mu_0']  # prior sample mean
        self.nu_0 = params['nu_0']  # prior "observations that make the prior mean"
        self.alpha_0 = params['alpha_0']  # prior IG shape
        self.beta_0 = params['beta_0']  # prior IG scale
        self.w = params['w']

        self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]
        self.action_var = [np.zeros(num_states)] * num_acts

    def update_stats(self, x, x_next, val, gamma):
        self.n[x] += 1
        m1 = val + gamma * self.B[x_next, 0]
        m2 = np.square(val) + 2 * gamma * val * self.B[x_next, 0] + np.square(
            gamma) * ((self.B[x_next, 1] + 1) / self.B[x_next, 1]) * (
                self.B[x_next, 3] / (self.B[x_next, 2] - 1) + np.square(self.B[x_next, 0]))

        self.B[x, 0] = (self.nu_0 * self.mu_0 + self.n[x] * m1) / (
            self.nu_0 + self.n[x])
        self.B[x, 1] = self.nu_0 + self.n[x]
        self.B[x, 2] = self.alpha_0 + (0.5 * self.n[x])
        self.B[x, 3] = self.beta_0 + 0.5 * self.n[x] * (
            m2 - np.square(m1)) + 0.5 * self.n[x] * (self.nu_0 * np.square(
                m1 - self.mu_0)) / (self.nu_0 + self.n[x])


    def sample(self, x, n, use_stddev=False):
        mu, nu, alpha, beta = self.B[x, :]
        scale = np.square(self.w) * max(0, beta * (nu + 1)/(nu * alpha))
        df = 2 * alpha
        try:
            # q_sa = t.rvs(df=df, loc=mu, scale=scale, size=n) #before Poster
            q_sa = t.ppf(q=self.q, df=df, loc=mu, scale=scale) #for Poster only
        except Exception as e:
            print(e)
            print(scale)
            print(mu, nu, alpha, beta)
            exit()
        return q_sa
