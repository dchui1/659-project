import numpy as np
import math

class BayesianApproximator():

    def __init__(self, state_dimensions, num_acts):

        self.dimensions = state_dimensions
        self.num_actions = num_acts
    #     self.B = np.zeros(np.prod(state_dimensions) + [num_acts])

    def update_stats(self, s, a, val=0.0):
        raise NotImplementedError

    def sample(self, s, a, n):
        raise NotImplementedError



class TabularBayesianApproximation(BayesianApproximator):
  def __init__(self, state_dimensions, num_acts):
    super().__init__(state_dimensions, num_acts)
    self.B = np.zeros([np.prod(self.dimensions)] + [self.num_actions, 4])
    print("Initiating tabular bayes with %d states and %d actions"%(np.prod(state_dimensions), num_acts))

    # prior sample mean
    # prior "observations to make that mean"
    # prior "observations to make our variance" # try to make it hard to reduce this
    # prior sum of square errors (proportional to initial sample variance)
    self.B[:, :] = [1, 1, 1, 4]

  def update_stats(self, s, a, val=0.0): # the default of the new value is 0 for exploration bonuses

    index = self.getIndex(s)
    mu, nu, alpha, beta = self.B[index, a]
    newValues = [0] * 4
    newValues[0] = (nu * mu + val) / (nu + 1)
    newValues[1] = nu + 1
    newValues[2] = alpha + 1.0/2.0
    newValues[3] = (nu / (nu + 1.0)) * math.pow((val - mu), 2.0) / 2.0
    self.B[index, a] = newValues

  def sample(self, s, a, n):
    index = self.getIndex(s)
    mu, nu, alpha, beta = self.B[index, a]
    variance = beta / ((alpha - 1.0) * nu)
    # don't add the mean here so we do not double count for the reward
    one_stdev = np.sqrt(variance)
    return [ one_stdev ]

  def getIndex(self, s):
      i = sum([a*b for a,b in zip(s[:-1], s[:-1])]) -1 + s[-1]
      return i
