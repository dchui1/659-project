import numpy as np
from agents.TabularQ import TabularQ
from utils.TabularQApproximation import TabularQApproximation
from utils.math import argMax
class BayesianQLearningTabular(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.gamma = params['gamma']
        self.bayesianQ = TabularQApproximation(state_shape, num_acts, self.gamma, params)

    def learn(self, s, sp, r, a, gamma):
        x = self.getIndex(sp) + (a * self.num_states)
        # self.bayesianQ.update_stats(x, r)
        # sampledValue = self.bayesianQ.sample(x, 100)
        # self.maxAction(sp)
        # s_idx = self.getIndex(s)
        # self.Q[s_idx, a] = self.Q[s_idx, a] + sampledValue

    # def q_sample(self, s, a):
    #     x = self.getIndex(s) + (a * self.num_states)
    #     return self.bayesianQ.sample(x, 100)

    def maxAction(self, s):
        act_vals = [self.bayesianQ.sample(self.getIndex(s) + (a * self.num_states), 100) for a in range(self.num_acts)]
        print("Action values for s", s, act_vals)
        move = argMax(act_vals)
        print("Move", move)
        return move
