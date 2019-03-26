import numpy as np
from agents.TabularQ import TabularQ
from utils.TabularQApproximation import TabularQApproximation
from utils.math import argMax
class BayesianQLearningTabular(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.gamma = params['gamma']
        self.bayesianQ = TabularQApproximation(state_shape, num_acts, self.gamma)

    def learn(self, s, sp, r, a, gamma):
        x = self.getIndex(s) + (a * self.num_states)
        ap = self.maxAction(sp)
        x_next = self.getIndex(sp) + (ap * self.num_states)
        self.bayesianQ.update_stats(x, x_next, r, gamma)

    def maxAction(self, s):
        self.act_vals = [self.bayesianQ.sample(self.getIndex(s) + (a * self.num_states), 1) for a in range(self.num_acts)]
        move = argMax(self.act_vals)
        return move
