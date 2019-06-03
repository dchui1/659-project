import numpy as np
import tensorflow as tf
import math
import scipy
from scipy.stats.distributions import t
import matplotlib.pyplot as plt
from src.bayesian_inference.tf_supervised_inference.distributions import T, MultivariateNormalInverseGamma, InverseGamma, MultivariateNormal
from src.bayesian_inference.tf_supervised_inference.linear_model import LinearModel


class BayesianApproximator():
    def __init__(self, state_dimensions, num_acts):

        self.dimensions = np.array(state_dimensions)
        self.num_actions = num_acts

    def update_stats(self, x, val=0.0):
        raise NotImplementedError

    def sample(self, x, n):
        raise NotImplementedError

class TabularBayesianApproximation(BayesianApproximator):
    def __init__(self, state_dimensions, num_acts, params={"mu_0": 1.0, "nu_0": 1.0, "alpha_0": 0.1, "beta_0": 1.0}):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))
        self.mu_0 = params["mu_0"]  # prior sample mean
        self.nu_0 = params["nu_0"]  # prior "observations that make the prior mean"
        self.alpha_0 = params["alpha_0"]  # prior IG shape
        self.beta_0 = params["beta_0"]  # prior IG scale
        self.q = params["q"]
        self.w = params["w"]
        self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]
        # To update beta, we need to keep track of the following 3 values for each (s,a)
        self.empirical_mean = np.zeros(num_states * num_acts)
        self.n = np.zeros(num_states * num_acts)  # local count for each (s, a) pair
        self.local_sum_sq = np.zeros(num_states * num_acts)

    def update_stats(self, x, val=0.0):
        self.n[x] += 1  # local count
        self.empirical_mean[x] += (val - self.empirical_mean[x]) / self.n[x]
        self.local_sum_sq[x] += np.square(val)

        self.B[x, 0] = (self.nu_0 * self.mu_0 + self.n[x] * self.empirical_mean[x]) / (self.nu_0 + self.n[x])
        self.B[x, 1] = self.nu_0 + self.n[x]
        self.B[x, 2] = self.alpha_0 + self.n[x] / 2


        sum_sq_residuals = self.local_sum_sq[x] - self.n[x] * np.square(self.empirical_mean[x])
        prior_residual_multiplier = ((self.n[x] * self.nu_0) / (self.n[x] + self.nu_0))
        prior_residual_sq = 0.5 * np.square(self.empirical_mean[x] - self.mu_0)
        self.B[x, 3] = self.beta_0 + 0.5 * sum_sq_residuals + prior_residual_multiplier * prior_residual_sq

    def sample(self, x, use_stddev=False):
        mu, nu, alpha, beta = self.B[x, :]
        scale = np.square(self.w) * max(0, beta * (nu + 1) / (alpha * nu)) # Make sure that the scale is >= 0
        df = 2 * alpha
        try:
            r = t.ppf(q=self.q, df=df, loc=mu, scale=scale)
        except:
            print(scale)
            exit()
        b = self.w * (r - mu)
        self.b = b
        return b
