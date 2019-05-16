import numpy as np
from src.agents.LinearQ import LinearQ
from src.bayesian_inference.bayesianapproximator import TDistBayesianApproximation

class TDistRLinearQ(LinearQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        # create mnig_prior: whose parameters come through params
        self.rewardApprox = TDistBayesianApproximation(state_shape, num_acts, params)
        self.num_samples = params['samples']

    def update(self, s, sp, r, a, done):
        x = self.tc.representation(s, a)
        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(self.num_samples)
        bonus = np.max(samples) - np.mean(samples)
        super().update(s, sp, r + bonus, a, done)
