import numpy as np
import math
import scipy
from scipy.stats import t
import matplotlib.pyplot as plt

class BayesianApproximator():

    def __init__(self, state_dimensions, num_acts):

        self.dimensions = np.array(state_dimensions)
        self.num_actions = num_acts

    def update_stats(self, x, val=0.0):
        raise NotImplementedError

    def sample(self, x, n):
        raise NotImplementedError

class TabularBayesianApproximation(BayesianApproximator):
  def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))
        # prior sample mean
        # prior "observations to make that mean"
        # prior "observations to make our variance" # try to make it hard to reduce this
        # prior sum of square errors (proportional to initial sample variance)
        self.B[:] = [1, 1, 2, 1]
        self.action_var = [np.zeros(num_states)] * num_acts

  def update_stats(self, x, val=0.0): # the default of the new value is 0 for exploration bonuses
    mu, nu, alpha, beta = self.B[x, :]
    self.B[x, 0] = (nu * mu + val) / (nu + 1)
    self.B[x, 1] = nu + 1
    self.B[x, 2] = alpha + 1.0/2.0
    self.B[x, 3] = (nu / (nu + 1.0)) * math.pow((val - mu), 2.0) / 2.0

  def sample(self, x, n, use_stddev=False):
    mu, nu, alpha, beta = self.B[x, :]
    scale = beta * (nu + 1)/(alpha * nu)
    df = 2 * alpha
    r = t.rvs(df=df, loc=mu, scale=scale, size=n)
    bonus = max(np.append(r, 0.0)) - np.average(r)
    # mean, var, skew, kurt = t.stats(df=df, loc=mu, scale=scale, moments='mvsk')
    return [bonus]
