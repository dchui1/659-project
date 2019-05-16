import numpy as np
from src.agents.LinearQ import LinearQ
from src.bayesian_inference.BNNApproximation import BNNApproximation

class BLRLinearQ(LinearQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.dimensions = state_shape
        self.num_actions = num_acts
        log_divergence_weight = params['log_divergence_weight']
        prior_stddev = params['prior_stddev']
        blr_alpha = params['blr_alpha']
        self.rewardApprox = BNNApproximation(state_shape, num_acts,
                                log_divergence_weight, prior_stddev, blr_alpha)
        self.rewardSamples = 10
        self.epochs = params['epochs']

    def update(self, s, sp, r, a, done):
        x = self.tc.representation(s, a)
        self.rewardApprox.update_stats(x, r, epochs=self.epochs)
        samples = self.rewardApprox.sample(x, self.rewardSamples)

        bonus = np.max(samples) - np.mean(samples)
        super().update(s, sp, r + bonus, a, done)
