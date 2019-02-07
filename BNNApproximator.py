import numpy as np
import BNN
class BNNApproximation(BayesianApproximator):

    def __init__(self, state_dimensions, num_acts):
        super().__init__(state_dimensions, num_acts)
        self.kernel = KernelPrior()
        self.bnn = self.kernel.new_bnn()


    def update_stats(self, s, a, val=0.0):
        raise NotImplementedError

    def sample(self, s, a, n):
        raise NotImplementedError
