import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# some the research2018 repo can be accessed from https://github.com/dmorrill10/research2018.git

class KernelPrior:
    def __init__(self, stddev=1):
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


def new_bnn(log_divergence_weight=-10, prior_stddev=1, residual_weight=0.1):
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
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                is_singular=False, loc_initializer=tf.zeros_initializer()),
            bias_divergence_fn=weighted_divergence_fn(log_divergence_weight))
    ])

if __name__ == "__main__":
    print(tf.__version__)
    sess = tf.InteractiveSession()
    tf.set_random_seed(42)
    np.random.seed(42)
    #training-data:
    x = np.array([[1., 2., 3.], [2., 3., 4.]], dtype=np.float32)
    y = np.array([1, 5], dtype=np.float32)
    # print("Shape of x", x.shape)
    y.shape[0]

    # run model
    f = new_bnn()
    f.compile(optimizer=tf.keras.optimizers.SGD(1e-1), loss=tf.losses.huber_loss)
    history = f.fit(x, y, batch_size=32, epochs=1000, verbose=1)#, class_weight=W)
    # print("history", history)
    #
    # print("Finished compiling")
    # print(type(f))
    # print(type(x))

    # Evaluation:
    prediction_train = f(x)
    predictions_list_trainset = []
    for i in range(100):
        y = sess.run(
            prediction_train
        )
        predictions_list_trainset.append(y)

    predictions_list_trainset = np.asarray(predictions_list_trainset)

    print("Predictions shape", predictions_list_trainset.shape)
    print("Predictions mean", predictions_list_trainset.mean())
    print("Predictions shape", predictions_list_trainset.std(0, ddof=1))

    #test-data:
    X_test = np.array([[45., 10., 10.], [500., 676., 2000.]], dtype=np.float32)

    prediction_test = f(X_test)
    predictions_list_testset = []
    for i in range(100):
        x = sess.run(
        prediction_test
        )
        predictions_list_testset.append(x)
    predictions_list_testset = np.asarray(predictions_list_testset)
    print("Predictions")
    print(predictions_list_testset.shape)
    print((predictions_list_testset.mean(0)).mean())
    print((predictions_list_testset.std(0, ddof=1)).mean())
