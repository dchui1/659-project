import numpy as np
from BNN import new_bnn
import tensorflow as tf
from bayesianapproximator import BayesianApproximator

class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts,
                    log_divergence_weight, prior_stddev, residual_weight,
                    blr_alpha):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = new_bnn(log_divergence_weight, prior_stddev, residual_weight)
        self.bnn.compile(optimizer=tf.keras.optimizers.SGD(blr_alpha),
            loss=tf.losses.huber_loss)


    def update_stats(self, x, val=0.0, batch_size = 1, epochs = 10):
        # s = x[:-1]
        # a = x[- 1]
        # if self.dimensions.shape != s.shape or not a <= self.num_actions:
        #     raise ValueError("Invalid value to update stats", s, a, val)

        # print("State action", s, a)
        # input_vector = np.concatenate((s, a))
        # print("The input vector shape", input_vector.shape)
        # print("The input vector", input_vector)


        self.bnn.fit(np.array([[x]]), np.array([[val]]), batch_size=batch_size, epochs=epochs, verbose=0)


    def sample(self, x, n):
        # s = x[:-1]
        # a = x[len(x) - 1]
        # if self.dimensions.shape != s.shape or not a <= self.num_actions:
        #     raise ValueError("Invalid value to sample", s, a)
        #
        # input_vector = np.concatenate((s, a))


        predictions = [self.bnn.predict(np.array([[x]], dtype=np.float32)) for i in range(n)]
        return predictions
