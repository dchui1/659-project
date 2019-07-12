# This is a modified version of BayesianQLearningTabular which
#     will be used to get results for the DLRLSS experiment.
#     The DLRSS experiment will be about placement of value bonus
#     in the action selection vs the action-value update function

import numpy as np
from src.agents.TabularQ import TabularQ
from src.bayesian_inference.TabularQApproximation import TabularQApproximation
from src.utils.math import argMax

class PosterBayesianQLearningTabular(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        print(params)
        super().__init__(state_shape, num_acts, params)
        self.gamma = params['gamma']
        self.bayesianQ = TabularQApproximation(state_shape, num_acts, self.gamma, params)
        self.percentile = params['percentile']

    def learn(self, s, sp, r, a, gamma):
        x = self.getIndex(s) + (a * self.num_states)
        ap = self.maxAction(sp)
        x_next = self.getIndex(sp) + (ap * self.num_states)
        self.bayesianQ.update_stats(x, x_next, r, gamma)

    def maxAction(self, s):
        # we sample 100 for each
        self.act_vals = [self.bayesianQ.sample(self.getIndex(s) + (a * self.num_states), 100) for a in range(self.num_acts)]
        # get the nth percentie
        self.act_vals = np.sort(self.act_vals, axis=1)[:,self.percentile-1]
        move = argMax(self.act_vals)
        return move
