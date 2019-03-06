import numpy as np
from BNN import BNN
import tensorflow as tf
from bayesianapproximator import BayesianApproximator

class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = BNN(state_dimensions * num_acts)

    def update_stats(self, x, val=0.0, batch_size = 1, epochs = 10):
        self.bnn.fit(x, val, epochs)


    def sample(self, x, n):
        return self.bnn.sample(x, n)
