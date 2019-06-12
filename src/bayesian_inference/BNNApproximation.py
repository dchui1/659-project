import numpy as np
from src.bayesian_inference.BNN import BNN
import tensorflow as tf
from src.bayesian_inference.bayesianapproximator import BayesianApproximator
from functools import reduce
import math

class BNNApproximation(BayesianApproximator):
    #BNN is

    def __init__(self, state_dimensions, num_acts, params):
        super().__init__(state_dimensions, num_acts)
        self.session = tf.InteractiveSession()
        self.bnn = BNN(reduce((lambda x, y: x * y), state_dimensions) * num_acts, params)
        self.q = params["q"]
        self.w = params["w"]

    def update_stats(self, x, val=0.0, batch_size = 1, epochs = 10):
        val = np.array([val])
        self.bnn.fit(x, val, epochs)

    def sample(self, x, n):
        array_of_means = self.bnn.sample(x, n)
        b = self.w * (array_of_means[math.floor(n*self.q)] - np.mean(array_of_means))
        # print("bonus ", b)
        return b
