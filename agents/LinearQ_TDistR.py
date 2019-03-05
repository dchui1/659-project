import numpy as np
from agents.LinearQ import LinearQ
from bayesianapproximator import TDistBayesianApproximation


class TDistRLinearQ(LinearQ):
    def __init__(self, state_shape, num_acts, params): # are these the initial parameters?
        super().__init__(state_shape, num_acts, params)
        # create mnig_prior: whose parameters come through params
        self.rewardApprox = TDistBayesianApproximation(state_shape, num_acts, params) # an instance of the TDist ... class
        self.num_samples = params['samples']

    def update(self, s, sp, r, a, done):
        x = self.tc.representation(s, a)
        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(self.num_samples)
        bonus = max(samples) - np.mean(samples)
        super().update(s, sp, r + bonus, a, done)
