import numpy as np
from agents.TabularQ import TabularQ
from BNNApproximation import BNNApproximation

class BnnRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)

        self.rewardApprox = BNNApproximation(state_shape, num_acts)
        self.epsilon = 0.05
        self.rewardSamples = 10

    def update(self, s, sp, r, a, done):
        x = np.concatenate([s,[a]])
        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(x, self.rewardSamples)

        bonus = np.max(samples) - np.mean(samples)
        # print("B bonus", bonus)
        super().update(s, sp, r, bonus, a, done)
