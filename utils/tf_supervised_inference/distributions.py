import tensorflow as tf
import numpy as np
from itertools import product
from utils.tf_supervised_inference.linear_model import LinearModel
from scipy.stats import t
'''
In order to create the T-distribution which will represent the Predicted Reward
Model, we need a prior MNIG distribution.
The MNIG is defined via an IG and a Normal distribution.

Since we want a high variance for the "prior" T-dist, then the scale this T-dist
must be > 1. Moreover, since T.scale = Normal.covariance * IG.scale / IG.shape,
then we want the IG.scale to be high and the IG.shape to be low.
Also, we need Normal.precision to be low (since the precision is the inverse of
the variance).

As for the "prior" T.mean, we want this value to be close to zero, therefore the
prior Normal.mean will be set to zero.

Putting all the above in practice will allow us to create a prior Multivariate
Normal and a prior IG distribution, which will generate a reasonable prior MNIG
distribution.

'''


class T(object):
    def __init__(self, mnig_prior):
        self.mnig_prior = mnig_prior

    def distribution(self):
        df = 2 * self.mnig_prior.ig_prior.shape
        mu = tf.transpose(self.mnig_prior.normal_prior.means)
        scale = self.mnig_prior.normal_prior.covariance() * (
            self.mnig_prior.ig_prior.scale / self.mnig_prior.ig_prior.shape)
        return tf.contrib.distributions.StudentT(df, mu, scale)

    def sample(self, num_samples=1):
        return tf.transpose(self.distribution().sample(num_samples))

    def next(
            self, x, y
    ):  # "x" represents the (s, a) pair, "y" represents the observed reward
        mnig_posterior = self.mnig_prior.next(x, y)
        return self.__class__(mnig_posterior)


class InverseGamma(object):
    @classmethod
    def from_log_mode_and_log_shape(cls, log_mode, log_shape):
        shape = tf.exp(log_shape)
        log_adjusted_shape = tf.log(shape + 1)
        scale = tf.exp(log_mode + log_adjusted_shape)
        return cls(shape, scale)

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self._inverse_distribution = tf.distributions.Gamma(
            self.shape, self.scale)

    def sample(self):
        return 1.0 / self._inverse_distribution.sample()

    def next(self, n, sse_estimate):
        return self.__class__(self.shape + n / 2.0,
                              self.scale + sse_estimate / 2.0)

    def mode(self):
        return self.scale / (self.shape + 1)

    def mean(self):
        return self.scale / (self.shape - 1) if self.shape > 1 else None

    def variance(self):
        return (self.mean()**2.0 / (self.shape - 1.0)
                if self.shape > 2 else None)


class MultivariateNormal(object):
    @classmethod
    def from_shared_mean_and_log_precision(cls, mean, log_precision, num_dims):
        precision = tf.exp(log_precision)
        return cls(
            tf.fill([num_dims, 1], mean / precision),
            precision * tf.eye(num_dims))

    def __init__(self, unscaled_means, precision, normal_prior=None):
        unscaled_means = tf.convert_to_tensor(unscaled_means)
        precision = tf.convert_to_tensor(precision)
        if normal_prior is None:
            self.precision = precision
        else:
            self.precision = normal_prior.precision + precision
            unscaled_means += normal_prior.weighted_precision_sums

        L = tf.cholesky(self.precision)
        self.means = tf.cholesky_solve(L, unscaled_means)

        self.covariance_scale = tf.matrix_triangular_solve(
            L, tf.eye(L.shape[0].value))

        self._weighted_precision_sums = None
        self._quadratic_form = None

    def distribution(self, scale=1.0):
        return tf.contrib.distributions.MultivariateNormalTriL(
            tf.transpose(self.means), scale * self.covariance_scale)

    @property
    def weighted_precision_sums(self):
        if self._weighted_precision_sums is None:
            self._weighted_precision_sums = self.precision @ self.means
        return self._weighted_precision_sums

    @property
    def quadratic_form(self):
        if self._quadratic_form is None:
            self._quadratic_form = tf.matmul(
                self.means, self.weighted_precision_sums, transpose_a=True)
        return self._quadratic_form

    def sample(self, scale=1.0):
        return tf.transpose(self.distribution(scale).sample())

    def next(self, weighted_feature_sums, empirical_precision):
        return self.__class__(
            weighted_feature_sums, empirical_precision, normal_prior=self)

    def covariance(self):
        return self.covariance_scale @ tf.transpose(self.covariance_scale)

    def maximum_a_posteriori_estimate(self):
        return LinearModel(self.means)


class ImproperMultivariateNormal(object):
    @classmethod
    def from_shared_mean_and_log_precision(cls, mean, log_precision, num_dims):
        precision = tf.exp(log_precision)
        return cls(
            tf.constant(mean, shape=[num_dims, 1]),
            precision * tf.eye(num_dims))

    def __init__(self, means, precision):
        self.means = means
        self.precision = precision
        self.weighted_precision_sums = self.precision @ self.means
        self.quadratic_form = tf.matmul(
            self.means, self.weighted_precision_sums, transpose_a=True)

    def next(self, weighted_feature_sums, empirical_precision):
        return MultivariateNormal(
            weighted_feature_sums, empirical_precision, normal_prior=self)


class MultivariateNormalInverseGamma(object):
    def __init__(self, normal_prior, ig_prior):
        self.normal_prior = normal_prior
        self.ig_prior = ig_prior

    def sample(self):
        return self.normal_prior.sample(self.ig_prior.sample())

    def next(self, x, y):
        x = tf.convert_to_tensor(x, dtype=np.float32)
        y = tf.convert_to_tensor(y, dtype=np.float32)

        x_T = tf.transpose(x)
        weighted_feature_sums = x_T @ y
        empirical_precision = x_T @ x

        normal_posterior = self.normal_prior.next(weighted_feature_sums,
                                                  empirical_precision)

        yty = tf.matmul(y, y, transpose_a=True, name='yty')
        sse_estimate = tf.squeeze(self.normal_prior.quadratic_form + yty -
                                  normal_posterior.quadratic_form)
        ig_posterior = self.ig_prior.next(
            tf.cast(tf.shape(x)[0], tf.float32), sse_estimate)

        return self.__class__(normal_posterior, ig_posterior)

    def maximum_a_posteriori_estimate(self):
        return self.normal_prior.maximum_a_posteriori_estimate()


class MultivariateNormalWithFixedMeasurementError(object):
    def __init__(self, normal_prior, scale):
        self.normal_prior = normal_prior
        self.scale = scale

    def sample(self):
        return self.normal_prior.sample(self.scale)

    def next(self, x, y):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        x_T = tf.transpose(x)
        weighted_feature_sums = x_T @ y
        empirical_precision = x_T @ x

        normal_posterior = self.normal_prior.next(weighted_feature_sums,
                                                  empirical_precision)
        return self.__class__(normal_posterior, self.scale)

    def maximum_a_posteriori_estimate(self):
        return self.normal_prior.maximum_a_posteriori_estimate()


# Please ignore for now:
class BayesianLinearRegressionDistribution(object):
    @classmethod
    def all(cls,
            phi,
            y,
            mean_params=[0.0],
            log_precision_params=np.arange(-5.0, 2.0).astype('float32'),
            log_ig_mode_params=[-20.0],
            log_ig_shape_params=[-20.0]):
        phi = tf.convert_to_tensor(phi)
        num_features = phi.shape[1].value

        for mean, log_precision, log_ig_mode, log_ig_shape in product(
                mean_params, log_precision_params, log_ig_mode_params,
                log_ig_shape_params):

            if np.isinf(log_ig_shape):
                normal = ImproperMultivariateNormal.from_shared_mean_and_log_precision(
                    mean=mean,
                    log_precision=log_precision,
                    num_dims=num_features)
                prior = MultivariateNormalWithFixedMeasurementError(
                    normal, tf.exp(log_ig_mode))
            else:
                ig = InverseGamma.from_log_mode_and_log_shape(
                    log_ig_mode, log_ig_shape)
                normal = ImproperMultivariateNormal.from_shared_mean_and_log_precision(
                    mean=mean,
                    log_precision=log_precision,
                    num_dims=num_features)
                prior = MultivariateNormalInverseGamma(normal, ig)

            yield (
                cls(prior).train(phi, y),
                mean,
                log_precision,
                log_ig_mode,
                log_ig_shape
            )  # yapf:disable

    def __init__(self, prior):
        self.posterior = prior

    def train(self, phi, y):
        self.posterior = self.posterior.next(phi, y)
        return self

    def sample(self, n=1):
        return [LinearModel(self.posterior.sample()) for _ in range(n)]

    def maximum_a_posteriori_estimate(self):
        return self.posterior.maximum_a_posteriori_estimate()
