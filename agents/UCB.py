import numpy as np
from agents.TabularQ import TabularQ
from utils.bayesianapproximator import TabularBayesianApproximation

class UCB(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.c = params['c']

        self.N = np.zeros((np.prod(state_shape), num_acts))
        self.steps = 0


    def update(self, s, sp, r, a, done):
        s_idx = self.getIndex(s)
        sp_idx = self.getIndex(sp)

        self.steps += 1
        self.N[s_idx, a] += 1

        n = self.N[s_idx, a]
        bonus = self.c * np.sqrt(np.log(self.steps) / n)

        super().update(s, sp, r + bonus, a, done)
