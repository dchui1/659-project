import numpy as np
from scipy.stats import t
from utils.bayesianapproximator import TabularBayesianApproximation

class TabularQApproximation(TabularBayesianApproximation):

  def __init__(self, state_dimensions, num_acts, gamma):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))

        self.mu_0 = 1.0  # prior sample mean
        self.nu_0 = 1.0  # prior "observations that make the prior mean"
        self.alpha_0 = 1.1  # prior IG shape
        self.beta_0 = 2.0  # prior IG scale
        self.gamma = gamma

        self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]
        self.action_var = [np.zeros(num_states)] * num_acts

  def update_stats(self, x, val=0.0): # the default of the new value is 0 for exploration bonuses
    mu, nu, alpha, beta = self.B[x, :]
    m1 = val + self.gamma * mu
    m2 = np.square(val) + 2 * self.gamma * val * mu + np.square(self.gamma) * (((nu + 1) / nu) * (beta / (alpha - 1)) + np.square(mu))
    print("m1", m1)
    print("m2", m2)
    self.B[x, 0] = (nu * mu + m1) / (nu + 1)
    self.B[x, 1] = nu + 1
    self.B[x, 2] = alpha + 1.0/2.0
    self.B[x, 3] = beta + (m2 - np.square(m1)) * 0.5  + ((nu * np.square(m1 - mu)) / (2 * (nu + 1)))

  def sample(self, x, n, use_stddev=False):
    # print("Sample called for x", x)
    mu, nu, alpha, beta = self.B[x, :]
    scale = beta * (nu + 1)/(alpha * nu)
    df = 2 * alpha
    try:
        r = t.rvs(df=df, loc=mu, scale=scale, size=n)
    except Exception as e:
        print(e)
        print(scale)
        print(mu, nu, alpha, beta)
        exit()
    bonus = np.average(r)
    # print(mu, nu, alpha, beta)
    return bonus
    # print(r)
    # return bonus.max()
