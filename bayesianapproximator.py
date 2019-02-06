import numpy as np
class BayesianApproximator():

    def __init__(self, state_dimensions, num_acts):

        self.B = np.zeros(np.prod(state_dimensions) + [num_acts])

    def update_stats(self, s, a, val=0.0):
        raise NotImplementedError

    def sample(self, s, a, n):
        raise NotImplementedError



class TabularBayesianApproximation(BayesianApproximator):
  def __init__(self, num_states, num_acts):
    self.B = np.zeros(np.prod(num_states) + [num_acts, 4])
    print("Initiating tabular bayes with %d states and %d actions"%(np.prod(num_states), num_acts))
    # prior sample mean
    # prior "observations to make that mean"
    # prior "observations to make our variance" # try to make it hard to reduce this
    # prior sum of square errors (proportional to initial sample variance)
    self.B[:, :] = [1, 1, 1, 4]

  def update_stats(self, s, a, val=0.0): # the default of the new value is 0 for exploration bonuses
    mu, nu, alpha, beta = self.B[s, a, :]
    self.B[s, a, 0] = (nu * mu + val) / (nu + 1)
    self.B[s, a, 1] = nu + 1
    self.B[s, a, 2] = alpha + 1.0/2.0
    self.B[s, a, 3] = (nu / (nu + 1.0)) * math.pow((val - mu), 2.0) / 2.0

  def sample(self, s, a, n):
    mu, nu, alpha, beta = self.B[s, a, :]
    variance = beta / ((alpha - 1.0) * nu)
    # don't add the mean here so we do not double count for the reward
    one_stdev = np.sqrt(variance)
    return [ one_stdev ]
