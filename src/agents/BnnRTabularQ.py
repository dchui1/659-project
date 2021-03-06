import numpy as np
from src.agents.TabularQ import TabularQ
from src.bayesian_inference.BNNApproximation import BNNApproximation

class BnnRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.dimensions = state_shape
        self.num_actions = num_acts
        self.rewardApprox = BNNApproximation(state_shape, num_acts)
        self.rewardSamples = 100
        self.epochs = params['epochs']

    def update(self, s, sp, r, a, done):
        # x = np.concatenate((self.convert_state(s),self.convert_action(a)))
        x = self.generate_input(s, a)
        self.rewardApprox.update_stats(x.flatten(), np.array([r]), epochs=self.epochs)
        samples = self.rewardApprox.sample(x.flatten(), self.rewardSamples)

        # If the max doesn't work out, we could also consider taking quantiles
        quantile_idx = 3 * (self.rewardSamples // 4) # get the index of the 3rd quantile
        bonus = samples[quantile_idx] - np.mean(samples)

        super().update(s, sp, r + bonus, a, done)

    def generate_input(self, s, a):

        input = np.zeros(self.dimensions + [self.num_actions])

        x = s[0]
        y = s[1]
        input[x][y][a] = 1
        return input

    def convert_state(self, s):
        sparse_vectors = []
        for i in range(len(self.dimensions)):
            arr = np.zeros(self.dimensions[i])
            arr[s[i]] = 1
            sparse_vectors.append(arr)
        return np.concatenate(sparse_vectors)

    def convert_action(self, a):
        arr = np.zeros(self.num_actions)
        arr[a] = 1
        return arr
