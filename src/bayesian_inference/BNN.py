import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util
tfd = tfp.distributions
# some the research2018 repo can be accessed from https://github.com/dmorrill10/research2018.git

"""
Creates a learnable variational normal distribution
Both the mean and variance are parameterized and are learnable by tensorflow
The variance is constrained to be non-negative
"""
def VariationalParameter(name, shape):
    means = tf.get_variable(name+'_mean', initializer=0.1 * tf.ones([1]), constraint=tf.keras.constraints.NonNeg())
    stds = tf.get_variable(name+'_std', initializer=1 * tf.ones([1]))
    return tfd.Normal(loc=means, scale=stds)

"""
Creates a *non*-learnable vector of normal distributions
This is used to regularize the weights
Weights will be pulled towards this distribution on each step
"""
class KernelPrior:
    def __init__(self, stddev):
        self.stddev = stddev

    def output(self, dtype, shape, name, trainable, add_variable_fn):
        dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(self.stddev)) #prior mean is zeros
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

class BNN:
    def __init__(self, features, params={'prior_mean_hidden_layer': -1e-5,
        'prior_stddev_hidden_layer': 1e-6,
        'prior_stddev_outer_layer': 1e-8}):
        self.features = features
        # Inputs to the tensorflow graph. X will be our phi(S, A), Y will be our reward
        self.X = tf.placeholder(tf.float32, [None, features])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.hidden_layer_mean = params['prior_mean_hidden_layer']
        self.hidden_layer_stddev = params['prior_stddev_hidden_layer']
        self.outer_layer_sttdev = params['prior_stddev_outer_layer']

        # Should be expandable to a deep network by adding more layers
        # Can add dense flipout layers for fully bayesian or could add simple dense or convolutional layers
        # to project into a smaller feature space before doing full distributions (would be more computationally efficient)
        self.layers = tf.keras.Sequential([
            tfp.layers.DenseFlipout(
                # one output for estimating the reward
                1,
                # the _prior_ distribution over our weights (even though it says posterior, it is the prior in the bayes rule sense)
                # this creates a vector of learnable independent normal distributions
                kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
                    # initialize the mean of the normal distributions randomly so that the means are slightly negative (pessimistic init)
                    loc_initializer=tf.random_normal_initializer(mean=self.hidden_layer_mean, stddev=self.hidden_layer_stddev) # prior mean and stddev of nodes in hidden layer
                ),
                # regularize our weights by pulling them towards a N(0, 1e-8) distribution
                # cannot have a N(0, 0) distribution, so pull them towards something with no variance
                kernel_prior_fn=KernelPrior(self.outer_layer_sttdev).output, # prior stddev over y's (outputs, in our case th rewards)
                # Don't use a bias weight here
                bias_posterior_fn=None, # set to None to keep everything local (local variance over all features)
            )
        ])

        # make predictions by sampling weights from the posterior and multiplying phi(S, A)
        self.predictions = self.layers(self.X)
        # model the variance of the noise on Y with a learnable normal distribution
        std = VariationalParameter('noise_std', [1])
        # build the distribution over Y ~ N(W*phi(S, A), std)
        pred_dist = tfd.Normal(loc=self.predictions, scale=std.sample())

        # Build the loss function
        # get the log probability of observing this value of Y given our parameters: P(Y | theta)
        log_prob = pred_dist.log_prob(self.Y)
        # make sure this log probability isn't nan (bug in tensorflow when variance approaches 0. if it is nan, just set it to zero)
        non_nan = tf.where(tf.is_nan(log_prob), tf.zeros_like(log_prob), log_prob)
        # get the mean over the outputs (only 1 output for now so this isn't really necessary, but it is good to be generic)
        neg_log_prob = -tf.reduce_mean(non_nan)
        # The KL-divergence is what trains the variance over the weights, the neg_log_prob is the loss over the mean
        # The KL-divergence is added as a "regularizer" to the layers as a hack to make this work with the tensorflow infrastructure (that's how tfp works)
        kl_div = sum(self.layers.losses)
        # the ELBO loss is just the sum of the loss over the variance (kl-div) and the loss over the mean (neg_log_prob)
        elbo_loss = neg_log_prob + kl_div

        # minimize the loss using some optimizer (adam with small learning rate seems to work well)
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train = optimizer.minimize(elbo_loss)

        # initialize the tensorflow graph and get initial values of the weights
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init_op.run()

    # makes sure that the X passed in is the right shape and size
    # if we accidentally pass in a tensor or vector, this should handle that case
    def normalizeXShape(self, x):
        return x.reshape((math.ceil(x.size / self.features), self.features))

    def fit(self, x, y, epochs=1):
        x = self.normalizeXShape(x)
        y = y.reshape((len(y), 1))
        feed = {self.X: x, self.Y: y}
        for _ in range(epochs):
            self.train.run(feed_dict=feed)

    def sample(self, x, samples=1000):
        x = self.normalizeXShape(x)
        m = np.tile(x, [samples, 1])
        feed = {self.X: m}
        p = self.predictions
        s = np.array(p.eval(feed_dict=feed)).flatten()
        s.sort()
        return s

if __name__ == "__main__":
    print(tf.__version__)
    sess = tf.InteractiveSession()
    tf.set_random_seed(42)
    np.random.seed(42)
    #training-data:
    x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y = np.array([[0], [0], [0]], dtype=np.float32)

    # run model
    f = BNN(3)
    s = f.sample(np.array([[1., 0., 0.]]))
    # print(s)
    print("Untrained mean and var", np.mean(s), np.std(s)**2)
    print("Weights", f.layers.get_weights())
    f.fit(x, y, epochs=100)
    s = f.sample(np.array([[1., 0., 0.]]))
    print(np.mean(s), np.std(s)**2)
    print(f.layers.get_weights())
