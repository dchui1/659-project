import numpy as np
from agents.LinearQ import LinearQ
from bayesianapproximator import TDistBayesianApproximation


class TDistRLinearQ(LinearQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        # create mnig_prior: whose parameters come through params
        self.rewardApprox = TDistBayesianApproximation(state_shape, num_acts, params)
        self.data = []
        self.reward_data = []

    def update(self, s, sp, r, a, done):
        x = self.tc.representation(s, a)\
            .array()
        x_mat = x.reshape(1, len(x))
        self.feature_vec = x.reshape(len(x), 1)
        r_mat = np.array([[r]])
        self.data.append(x_mat)
        self.reward_data.append(r_mat)
        self.rewardApprox.update_stats(x_mat, r_mat)
        samples = self.rewardApprox.sample(x_mat, self.params["num_samples"])
        bonus = np.max(samples) - np.mean(samples)
        super().update(s, sp, r + bonus, a, done)
