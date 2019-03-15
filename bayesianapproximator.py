import numpy as np
import tensorflow as tf
import math
import scipy
from scipy.stats import t
import matplotlib.pyplot as plt
from tf_supervised_inference.distributions import T, \
    MultivariateNormalInverseGamma, InverseGamma, MultivariateNormal
from tf_supervised_inference.linear_model import LinearModel


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
        ig_prior = InverseGamma.from_log_mode_and_log_shape(
            tf.log(params["ig_prior_mode"]), tf.log(params["ig_prior_shape"]))
        # prior shape and scale. A large scale means the data is more broad.
        num_dims = np.square(params["tiles"]) * params["tilings"] * num_acts
        normal_prior = MultivariateNormal.from_shared_mean_and_log_precision(
            params["normal_prior_mean"],
            params[
                "normal_prior_log_precision"],  # small precision = large variance
            num_dims=num_dims)
        self.mnig_prior = MultivariateNormalInverseGamma(
            normal_prior, ig_prior)
        self.T_distribution = T(self.mnig_prior)

    def update_stats(self, x, y):
        self.T_distribution = self.T_distribution.next(x, y)
        return self.T_distribution

    def sample(self, x, num_samples):
        weights = self.T_distribution.sample(num_samples)
        return tf.matmul(
            weights, x.astype('float32'), transpose_b=True).numpy()

# The following class implements a global IG distribution (univariate)
class TabularBayesianApproximation2(BayesianApproximator):
    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.alpha_0 = 0.1
        self.beta_0 = 1.0
        self.nu_0 = 1.0
        self.mu_0 = 0.0

        self.n = np.zeros(
            (num_states * num_acts))  #[0] * num_states * num_acts
        self.empirical_mean = np.zeros(
            (num_states * num_acts))  # [0] * num_states * num_acts
        self.global_sum_sq = 0.0

    def update_stats(self, x, val=0.0):
        self.n[x] += 1  # local count
        self.empirical_mean[x] += (val - self.empirical_mean[x]) / self.n[x]
        self.global_sum_sq += np.square(val)

    def posterior_global_beta(self):
        return self.beta_0 + 0.5 * (
            self.global_sum_sq -
            self.global_n() * np.square(self.global_empirical_mean())) + (
                self.global_n() * self.nu_0) / (self.nu_0 + self.global_n(
                )) * 0.5 * np.square(self.global_empirical_mean() - self.mu_0)

    def global_empirical_mean(self):
        return np.mean(self.empirical_mean)

    def posterior_mean(self, x):
        return (self.nu_0 * self.mu_0 +
                self.n[x] * self.empirical_mean[x]) / self.posterior_nu(x)

    def posterior_global_alpha(self):
        return self.alpha_0 + self.global_n() / 2

    def posterior_nu(self, x):
        return self.nu_0 + self.n[x]

    def global_n(self):
        return np.sum(self.n)

    def sample(self, x, n, use_stddev=False):
        mu = self.posterior_mean(x)
        nu = self.posterior_nu(x)
        alpha = self.posterior_global_alpha()
        scale = max(0, self.posterior_global_beta() * (nu + 1) / (alpha * nu))
        df = 2 * alpha
        try:
            r = t.rvs(df=df, loc=mu, scale=scale, size=n)
        except:
            print(scale)
            exit()
        bonus = np.maximum(r, 0.0) - np.average(r)
        return bonus

# The following class implements a local IG distribution (multivariate)
class TabularBayesianApproximation(BayesianApproximator):
    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        num_states = np.prod(state_dimensions)
        self.state_shape = state_dimensions
        self.alpha_0 = 0.1
        self.beta_0 = 1.0
        self.nu_0 = 1.0
        self.mu_0 = 0.0

        self.n = np.zeros(
            (num_states * num_acts))  #[0] * num_states * num_acts
        self.empirical_mean = np.zeros(
            (num_states * num_acts))  # [0] * num_states * num_acts
        self.local_sum_sq = np.zeros((num_states * num_acts))

    def update_stats(self, x, val=0.0):
        self.n[x] += 1  # local count
        self.empirical_mean[x] += (val - self.empirical_mean[x]) / self.n[x]
        self.local_sum_sq[x] += np.square(val)

    def posterior_local_beta(self, x):
        return self.beta_0 + 0.5 * (
            self.local_sum_sq[x] - self.n[x] * np.square(
                self.empirical_mean[x])) + (self.n[x] * self.nu_0) / (
                    self.nu_0 + self.n[x]
                ) * 0.5 * np.square(self.empirical_mean[x] - self.mu_0)

    def posterior_local_mean(self, x):
        return (self.nu_0 * self.mu_0 +
                self.n[x] * self.empirical_mean[x]) / self.posterior_nu(x)

    def posterior_local_alpha(self, x):
        return self.alpha_0 + self.n[x] / 2

    def posterior_nu(self, x):
        return self.nu_0 + self.n[x]

    def sample(self, x, n, use_stddev=False):
        mu = self.posterior_local_mean(x)
        nu = self.posterior_nu(x)
        alpha = self.posterior_local_alpha(x)
        scale = max(0, self.posterior_local_beta(x) * (nu + 1) / (alpha * nu))
        df = 2 * alpha
        try:
            r = t.rvs(df=df, loc=mu, scale=scale, size=n)
        except:
            print(scale)
            exit()
        bonus = np.maximum(r, 0.0) - np.average(r)
        return bonus.max()
    # def sample_without_updating(self, x, n, use_stddev=False):
