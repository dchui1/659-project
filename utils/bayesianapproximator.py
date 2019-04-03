import numpy as np
import tensorflow as tf
import math
import scipy
from scipy.stats import t
import matplotlib.pyplot as plt
from utils.tf_supervised_inference.distributions import T, MultivariateNormalInverseGamma, InverseGamma, MultivariateNormal
from utils.tf_supervised_inference.linear_model import LinearModel


class BayesianApproximator():
    def __init__(self, state_dimensions, num_acts):

        self.dimensions = np.array(state_dimensions)
        self.num_actions = num_acts

    def update_stats(self, x, val=0.0):
        raise NotImplementedError

    def sample(self, x, n):
        raise NotImplementedError


class TDistBayesianApproximation(BayesianApproximator):
    def __init__(self, state_dimensions, num_acts, params):
        super().__init__(state_dimensions, num_acts)
        ig_prior = InverseGamma(
            params["ig_prior_shape"], params["ig_prior_scale"]
        )  # prior shape and scale. A large scale means the data is more broad.
        num_dims = np.square(params["tiles"]) * params["tilings"] * num_acts
        normal_prior = MultivariateNormal.from_shared_mean_and_log_precision(
            params["normal_prior_mean"],
            params[
                "normal_prior_log_precision"],  # small precision = large variance
            num_dims=num_dims)
        self.mnig_prior = MultivariateNormalInverseGamma(
            normal_prior, ig_prior)
        self.distribution_dimension = np.prod(state_dimensions) * num_acts
        self.T_distribution = T(self.mnig_prior)

    def update_stats(self, x, y):
        self.T_distribution = self.T_distribution.next(x, y)
        return self.T_distribution

    def sample(self, num_samples):
        weights = self.T_distribution.sample(num_samples)
        return [
            LinearModel(self.posterior.sample()) for _ in range(num_samples)
        ]


class TabularBayesianApproximation(BayesianApproximator):
    def __init__(self, state_dimensions, num_acts, params={"mu_0":
1.0, "nu_0": 1.0, "alpha_0": 0.1, "beta_0": 1.0}):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.B = np.zeros((num_states * num_acts, 4))
        self.mu_0 = params["mu_0"]  # prior sample mean
        self.nu_0 = params["nu_0"]  # prior "observations that make the prior mean"
        self.alpha_0 = params["alpha_0"]  # prior IG shape
        self.beta_0 = params["beta_0"]  # prior IG scale

        self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]

        # To update beta, we need to keep track of the following 3 values for each (s,a)
        self.empirical_mean = np.zeros((num_states * num_acts))
        self.n = np.zeros(
            (num_states * num_acts))  # local count for each (s, a) pair
        self.local_sum_sq = np.zeros((num_states * num_acts))

    def update_stats(self, x, val=0.0):
        self.n[x] += 1  # local count
        self.empirical_mean[x] += (val - self.empirical_mean[x]) / self.n[x]
        self.local_sum_sq[x] += np.square(val)

        self.B[x, 0] = (self.nu_0 * self.mu_0 + self.n[x] *
                        self.empirical_mean[x]) / (self.nu_0 + self.n[x])
        self.B[x, 1] = self.nu_0 + self.n[x]
        self.B[x, 2] = self.alpha_0 + self.n[x] / 2
        self.B[x, 3] = (self.beta_0 + 0.5
            * (self.local_sum_sq[x] - self.n[x] * np.square(self.empirical_mean[x]))
            + (self.n[x] * self.nu_0) / (self.nu_0 + self.n[x])
            * 0.5 * np.square(self.empirical_mean[x] - self.mu_0))

    def sample(self, x, n, use_stddev=False):
        mu, nu, alpha, beta = self.B[x, :]
        scale = max(0, beta * (nu + 1) / (alpha * nu)) # Make sure that the scale is >= 0
        df = 2 * alpha
        try:
            r = t.rvs(df=df, loc=mu, scale=scale, size=n)
        except:
            print(scale)
            exit()
        bonus = np.maximum(r, 0.0) - np.average(r)
        return bonus.max()


# class TabularBayesianApproximation_one_step_update(BayesianApproximator):
#   def __init__(self, state_dimensions, num_acts):
#         super().__init__(state_dimensions, num_acts)
#         num_states = np.prod(state_dimensions)
#         self.state_shape = state_dimensions
#         self.B = np.zeros((num_states * num_acts, 4))
#         self.mu_0 = 1.0  # prior sample mean
#         self.nu_0 = 1.0  # prior "observations that make the prior mean"
#         self.alpha_0 = 0.1  # prior IG shape
#         self.beta_0 = 1.0  # prior IG scale
#
#         self.B[:] = [self.mu_0, self.nu_0, self.alpha_0, self.beta_0]
#         self.action_var = [np.zeros(num_states)] * num_acts
#
#   def update_stats(self, x, val=0.0): # the default of the new value is 0 for exploration bonuses
#     mu, nu, alpha, beta = self.B[x, :]
#     self.B[x, 0] = (nu * mu + val) / (nu + 1)
#     self.B[x, 1] = nu + 1
#     self.B[x, 2] = alpha + 1.0/2.0
#     self.B[x, 3] = beta + (nu / (nu + 1.0)) * 0.5 * np.square(val - mu)
#
#   def sample(self, x, n, use_stddev=False):
#     mu, nu, alpha, beta = self.B[x, :]
#     scale = beta * (nu + 1)/(alpha * nu)
#     df = 2 * alpha
#     try:
#         r = t.rvs(df=df, loc=mu, scale=scale, size=n)
#     except:
#         print(scale)
#         exit()
#     bonus = np.maximum(r, 0.0) - np.average(r)
#     return bonus.max()
