import numpy as np
from agents.TabularQ import TabularQ
from bayesianapproximator import TabularBayesianApproximation

class TabularRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts):
        super().__init__(state_shape, num_acts)
        self.rewardApprox = TabularBayesianApproximation(state_shape, num_acts)
        self.epsilon = 0.05

    def update(self, s, sp, r, a, done):
        self.rewardApprox.update_stats(s, a, r)
        samples = self.rewardApprox.sample(s, a, 1)
        bonus = samples[0] # wouldn't this raise a type error?
        super().update(s, sp, r + bonus, a, done)
