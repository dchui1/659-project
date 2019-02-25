import numpy as np
from agents.TabularQ import TabularQ
from bayesianapproximator import TabularBayesianApproximation

class TabularRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.rewardApprox = TabularBayesianApproximation(state_shape, num_acts)

    def update(self, s, sp, r, a, done):
        x = self.getIndex(s) + (a * self.num_states)
        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(x, 100)
        bonus = samples[0] # wouldn't this raise a type error?
        super().update(s, sp, r, bonus, a, done)
