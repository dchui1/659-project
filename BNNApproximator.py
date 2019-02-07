import numpy as np
from BNN import *
class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)

        self.bnn = new_bnn()
        #TODO refactor this back out into the BNN method
        f.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=tf.losses.huber_loss)
        history = f.fit(state_dimensions, num_acts, batch_size=32, epochs=1000, verbose=1)


    def update_stats(self, s, a, val=0.0):
        raise NotImplementedError

    def sample(self, s, a, n):
        raise NotImplementedError
