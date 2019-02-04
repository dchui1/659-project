import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# some the research2018 repo can be accessed from https://github.com/dmorrill10/research2018.git
import research2018.layers
from research2018.layers import ResConvolution2DFlipout

# Here, X is a numpy.nd array containing states.
# Each state is a 4 x 7 x 3 numpy.ndarray (4 x 7 cells, 1 RGB color per cell)

x_train_abs_max = np.abs(X).max(axis=0, keepdims=True).astype("float32")
x_train_abs_min = np.abs(X).min(axis=0, keepdims=True).astype("float32")
x_train_mean = np.abs(X).mean(axis=0, keepdims=True).astype("float32")
num_actions = len(ACTIONS)

def normalize_x(x):
    abs_x = tf.abs(x)
    num_rows = tf.shape(abs_x)[0]
    def tile(x):
        return tf.tile(x, [num_rows, 1, 1, 1])

    max_obs = tf.maximum(
        tf.reduce_max(abs_x, axis=0, keepdims=True), x_train_abs_max)
    min_obs = tf.minimum(
        tf.reduce_min(abs_x, axis=0, keepdims=True), x_train_abs_min)
    delta_obs = tile(max_obs - min_obs)
    return tf.where(
        tf.greater(delta_obs, 0.0), (x - x_train_mean) / delta_obs,
        tf.zeros_like(x))


class KernelPrior(object):
    def __init__(self, stddev=1):
        self.stddev = stddev

    def output(self, dtype, shape, name, trainable, add_variable_fn):
        scale = np.full(shape, self.stddev, dtype=dtype.as_numpy_dtype)
        dist = tfp.distributions.Normal(
            loc=tf.zeros(shape, dtype), scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)

    def conv(self, dtype, shape, name, trainable, add_variable_fn):
        dist = tfp.distributions.Normal(
            loc=tf.zeros(shape, dtype),
            scale=dtype.as_numpy_dtype(self.stddev))
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)


def weighted_divergence_fn(log_weight):
    def divergence_fn(pos, pri, _):
        return (
            tf.exp(float(log_weight)) * tf.reduce_mean(pos.kl_divergence(pri)))

    return divergence_fn


def new_bnn(filters=10, log_divergence_weight=-10, prior_stddev=1, residual_weight=0.1):
    '''
    Creates a new Bayesian neural network on images.

    Arguments:
    - filters: The number of convolutional filters in the two hidden convolutional layers. Defaults to 8 just because I saw another script use this many.
    - log_divergence_weight: The weight of the divergence penalty on each layer. Defaults to -3 since that worked best for me with MNIST.
    - prior_stddev: The standard deviation of the prior weight distributions. Defaults to 1 since that should probably be a good place to start. Might need to turn this up a lot though to get the layers to be more random, so you could set this as large as 20.
    - residual_weight: The weight on the residual term in the residual layers. Defaults to 0.1 since that worked best for me. You can set it to zero to make the layers non-residual.
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: normalize_x(tf.cast(x, tf.float32))), # normalizes features, and ensures have right type
        tfp.layers.Convolution2DFlipout(
            filters=filters,
            kernel_size=5,
            padding='SAME',
            activation=tf.nn.elu,
            kernel_prior_fn=KernelPrior(prior_stddev).conv,
            kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                is_singular=False, loc_initializer=tf.zeros_initializer()),
            bias_divergence_fn=weighted_divergence_fn(log_divergence_weight)),
        tf.keras.layers.AveragePooling2D(
            pool_size=[2, 2], strides=[2, 2], padding='SAME'), #averages each pair of cells in image
        ResConvolution2DFlipout(
            filters=filters,
            kernel_size=5,
            padding='SAME',
            activation=tf.nn.elu,
            residual_weight=residual_weight,
            kernel_prior_fn=KernelPrior(prior_stddev).conv,
            kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                is_singular=False, loc_initializer=tf.zeros_initializer()),
            bias_divergence_fn=weighted_divergence_fn(log_divergence_weight)),
        tf.keras.layers.AveragePooling2D(
            pool_size=[2, 2], strides=[2, 2], padding='SAME'),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            num_actions,
            kernel_prior_fn=KernelPrior(prior_stddev).output,
            kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                is_singular=False, loc_initializer=tf.zeros_initializer()),
            bias_divergence_fn=weighted_divergence_fn(log_divergence_weight))
    ])


f = new_bnn()
# def loss(Y, Y_hat):
#   print(Y.shape)
#   return tf.reduce_mean(W * 0.5 * tf.square(Y - Y_hat))

# Y is a S x A array (previously storing the bonus values for each (s, a)).
# W is a S x A array of indicator vectors. Each indicator vector has length |A|,
# and has a "1" in the position of the action taken at state s.


f.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=tf.losses.huber_loss)
history = f.fit(X, Y, batch_size=32, epochs=1000, verbose=1, class_weight=W)
