import numpy as np
from src.bayesian_inference.BNN import BNN
import tensorflow as tf
from src.bayesian_inference.bayesianapproximator import BayesianApproximator
from functools import reduce

class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = BNN(reduce((lambda x, y: x * y), state_dimensions)  * num_acts)

    def update_stats(self, x, val=0.0, batch_size = 1, epochs = 10):
        self.bnn.fit(x, val, epochs)


    def sample(self, x, n):
        return self.bnn.sample(x, n)