import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
# some the research2018 repo can be accessed from https://github.com/dmorrill10/research2018.git

def VariationalParameter(name, shape, constraint=None):
    """Generates variational distribution(s)"""
    means = tf.get_variable(name+'_mean', initializer=tf.ones([1]), constraint=constraint)
    stds = tf.get_variable(name+'_std', initializer=-1*tf.ones([1]))
    return tfd.Normal(loc=means, scale=tf.math.exp(stds))

class KernelPrior:
    def __init__(self, stddev):
        self.stddev = stddev

    def output(self, dtype, shape, name, trainable, add_variable_fn):
        scale = np.full(shape, self.stddev, dtype=dtype.as_numpy_dtype)
        dist = tfp.distributions.Normal(
            loc=tf.zeros(shape, dtype), scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)


def weighted_divergence_fn(log_weight):
    def divergence_fn(pos, pri, _):
        return (
            tf.exp(float(log_weight)) * tf.reduce_mean(pos.kl_divergence(pri)))
    return divergence_fn

class BNN:
    def __init__(self, features):
        self.X = tf.placeholder(tf.float32, [1, features])
        self.Y = tf.placeholder(tf.float32, [1,])

        self.layers = tf.keras.Sequential([
            tfp.layers.DenseFlipout(
                1,
                bias_posterior_fn=None,
                kernel_prior_fn=KernelPrior(0.01)
            ),
        ])

        self.predictions = self.layers(self.X)
        std = VariationalParameter('noise_std', [1], constraint=tf.keras.constraints.NonNeg())
        self.pred_dist = tfd.Normal(loc=self.predictions, scale=std.sample())

        neg_log_prob = -tf.reduce_mean(self.pred_dist.log_prob(self.Y))
        kl_div = sum(self.layers.losses)
        self.elbo_loss = neg_log_prob + kl_div

        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = self.optimizer.minimize(self.elbo_loss)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init_op.run()

    def fit(self, x, y, epochs=1):
        for _ in range(epochs):
            self.train.run(feed_dict={self.X: x, self.Y: y})
            print(self.elbo_loss.eval(feed_dict={self.X: x, self.Y: y}))

        print(self.layers.layers[0].get_weights())

    def sample(self, x, samples=5):
        y = self.pred_dist.sample(5)
        print(y.eval(feed_dict={self.X: x}))


def new_bnn(log_divergence_weight=-1, prior_stddev=0.5, residual_weight=0.1):

    '''
    Creates a new Bayesian neural network.

    Arguments:
    - log_divergence_weight: The weight of the divergence penalty on each layer.
        Defaults to -3 since that worked best for me with MNIST.
    - prior_stddev: The standard deviation of the prior weight distributions.
        Defaults to 1 since that should probably be a good place to start.
        Might need to turn this up a lot though to get the layers to be more
        random, so you could set this as large as 20.
    - residual_weight: The weight on the residual term in the residual layers.
        Defaults to 0.1 since that worked best for me. You can set it to zero
        to make the layers non-residual.
    '''
    return tf.keras.Sequential([
        tfp.layers.DenseFlipout(
            1,
            kernel_prior_fn=KernelPrior(prior_stddev).output,
            kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
            bias_posterior_fn=None
        )
    ])

if __name__ == "__main__":
    print(tf.__version__)
    sess = tf.InteractiveSession()
    tf.set_random_seed(42)
    np.random.seed(42)
    #training-data:
    x = np.array([[1, 0, 0]], dtype=np.float32)
    y = np.array([0], dtype=np.float32)

    # run model
    f = BNN(3)
    f.fit(x, y, epochs=100)
    f.sample(x)
    exit(0)
    # print("history", history)
    #
    # print("Finished compiling")
    # print(type(f))
    # print(type(x))

    # Evaluation:
    # predictions_list_trainset = []
    # for i in range(100):
    #     y = sess.run(
    #         prediction_train
    #     )
    #     predictions_list_trainset.append(y)

    # predictions_list_trainset = np.asarray(predictions_list_trainset)

    # print("Predictions shape", predictions_list_trainset.shape)
    # print("Predictions mean", predictions_list_trainset.mean())
    # print("Predictions shape", predictions_list_trainset.std(0, ddof=1))

    # #test-data:
    # X_test = np.array([[45., 10., 10.], [500., 676., 2000.]], dtype=np.float32)

    # prediction_test = f(X_test)
    # predictions_list_testset = []
    # for i in range(100):
    #     x = sess.run(
    #     prediction_test
    #     )
    #     predictions_list_testset.append(x)
    # predictions_list_testset = np.asarray(predictions_list_testset)
    # print("Predictions")
    # print(predictions_list_testset.shape)
    # print((predictions_list_testset.mean(0)).mean())
    # print((predictions_list_testset.std(0, ddof=1)).mean())
