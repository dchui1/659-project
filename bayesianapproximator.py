import numpy as np
import math
import scipy
from scipy.stats import t
import matplotlib.pyplot as plt

class BayesianApproximator():

    def __init__(self, state_dimensions, num_acts):

        self.dimensions = np.array(state_dimensions)
        self.num_actions = num_acts

    def update_stats(self, s, a, val=0.0):
        raise NotImplementedError

    def sample(self, s, a, n):
        raise NotImplementedError

class TabularBayesianApproximation(BayesianApproximator):
  def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states, num_acts, 4))
        # prior sample mean
        # prior "observations to make that mean"
        # prior "observations to make our variance" # try to make it hard to reduce this
        # prior sum of square errors (proportional to initial sample variance)
        self.B[:, :] = [1, 1, 2, 1]
        self.action_var = [np.zeros(num_states)] * num_acts

  def getIndex(self, s):
    return np.ravel_multi_index(s, self.state_shape)

  def update_stats(self, s, a, val=0.0): # the default of the new value is 0 for exploration bonuses
    s_idx = self.getIndex(s)
    mu, nu, alpha, beta = self.B[s_idx, a, :]
    self.B[s_idx, a, 0] = (nu * mu + val) / (nu + 1)
    self.B[s_idx, a, 1] = nu + 1
    self.B[s_idx, a, 2] = alpha + 1.0/2.0
    self.B[s_idx, a, 3] = (nu / (nu + 1.0)) * math.pow((val - mu), 2.0) / 2.0

  def sample(self, s, a, n, use_stddev=False):
    s_idx = self.getIndex(s)
    mu, nu, alpha, beta = self.B[s_idx, a, :]
    scale = beta * (nu + 1)/(alpha * nu)
    df = 2 * alpha
    r = t.rvs(df=df, loc=mu, scale=scale, size=1000)
    bonus = max(np.append(r, 0.0)) - np.average(r)
    mean, var, skew, kurt = t.stats(df=df, loc=mu, scale=scale, moments='mvsk')
    print(var)
    self.action_var[a][s_idx] = var
    # maybe we could check that the variance is -> 0 for each (s, a)
    if use_stddev:
        variance = beta / ((alpha - 1.0) * nu) # we could use calculated "variance", or "var"
        # don't add the mean here so we do not double count for the reward
        one_stdev = np.sqrt(variance)
        return [one_stdev]
    return [bonus]
