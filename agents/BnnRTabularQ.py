import numpy as np
from agents.TabularQ import TabularQ
from BNNApproximation import BNNApproximation

class BnnRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.dimensions = state_shape
        self.num_actions = num_acts
        log_divergence_weight = params['log_divergence_weight']
        prior_stddev = params['prior_stddev']
        blr_alpha = params['blr_alpha']
        self.rewardApprox = BNNApproximation(state_shape, num_acts,
                                log_divergence_weight, prior_stddev, blr_alpha)
        self.epsilon = 0.05
        self.rewardSamples = 10
        self.epochs = params['epochs']

    def update(self, s, sp, r, a, done):
        # x = np.concatenate((self.convert_state(s),self.convert_action(a)))
        x = self.generate_input(s, a)
        self.rewardApprox.update_stats(x, r, epochs=self.epochs)
        samples = self.rewardApprox.sample(x, self.rewardSamples)

        bonus = np.max(samples) - np.mean(samples)
        # print("B bonus", bonus, np.max(samples), np.mean(samples))
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
