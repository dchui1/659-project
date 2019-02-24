import numpy as np
from agents.LinearQ import LinearQ
from bayesianapproximator import TDistBayesianApproximation


class TDistRLinearQ(LinearQ):
    def __init__(self, state_shape, num_acts, params): # are these the initial parameters?
        super().__init__(state_shape, num_acts, params)
        # create mnig_prior: whose parameters come through params
        self.rewardApprox = TDistBayesianApproximation(state_shape, num_acts, params) # an instance of the TDist ... class

    def update(self, s, sp, r, a, num_samples, done):
        x = self.tc.representation(s, a)
        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(num_samples)
        bonus = max(samples) - np.mean(samples)
        super().update(s, sp, r + bonus, a, done)


# How to test this agent?
# agent = TDistRLinearQ((30, 30), 4, [{"alpha" : 0.1, "epsilon": 0.1,
#     "gamma": 0.9, "tiles" : 30, "tilings" : 1}])
