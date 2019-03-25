import numpy as np
from scipy.stats import t
from utils.bayesianapproximator import TabularBayesianApproximation

class TabularQApproximation(TabularBayesianApproximation):
    def __init__(self, state_dimensions, num_acts, gamma, params={'mu_0': 0.0, 'nu_0': 1.0, 'alpha_0': 1.0, 'beta_0': 0.1}):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))
        self.n = np.zeros((num_states * num_acts))

        self.mu_0 = params['mu_0']  # prior sample mean
        self.nu_0 = params['nu_0']  # prior "observations that make the prior mean"
        self.alpha_0 = params['alpha_0']  # prior IG shape
        self.beta_0 = params['beta_0']  # prior IG scale

        self.gamma = gamma

        self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]
        self.action_var = [np.zeros(num_states)] * num_acts
        # self.t_distribution_var = np.zeros((num_states * num_acts))

    def update_stats(self, x, val=0.0):
        # get the state, action index back from x
        self.n[x] += 1
        m1 = val + self.gamma * self.B[x, 0]
        m2 = np.square(val) + 2 * self.gamma * val * self.B[x, 0] + np.square(
            self.gamma) * ((self.B[x, 1] + 1) / self.B[x, 1]) * (
                self.B[x, 3] / (self.B[x, 2] - 1) + np.square(self.B[x, 0]))
        # print("m1", m1)
        # print("m2", m2)
        self.B[x, 0] = (self.nu_0 * self.mu_0 + self.n[x] * m1) / (
            self.nu_0 + self.n[x])
        self.B[x, 1] = self.nu_0 + self.n[x]
        self.B[x, 2] = self.alpha_0 + (0.5 * self.n[x])
        self.B[x, 3] = self.beta_0 + 0.5 * self.n[x] * (
            m2 - np.square(m1)) + 0.5 * self.n[x] * (self.nu_0 * np.square(
                m1 - self.mu_0)) / (self.nu_0 + self.n[x])

        # s_idx = np.ravel_multi_index(s, self.state_shape)
        self.t_distribution_var = self.B[x, 3] * (self.B[x, 1] + 1) / (
            self.B[x, 2] * self.B[x, 1])

    def sample(self, x, n, use_stddev=False):
        # print("Sample called for x", x)
        mu, nu, alpha, beta = self.B[x, :]
        scale = max(0, beta * (nu + 1) / (alpha * nu))
        df = 2 * alpha
        try:
            q_sa = t.rvs(df=df, loc=mu, scale=scale, size=1)
        except Exception as e:
            print(e)
            print(scale)
            print(mu, nu, alpha, beta)
            exit()
        # print(mu, nu, alpha, beta)
        return q_sa
        # print(r)
        # return bonus.max()
