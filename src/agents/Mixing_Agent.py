import numpy as np
from src.agents.TabularQ import TabularQ
from src.utils.math import argMax


class Mixing_Agent(TabularQ):
    def __init__(self, state_shape, num_acts, params):
      super().__init__(state_shape, num_acts, params)

    def update(self, s, sp, r, bonus_scalar, a, done):
      super().update(s, sp, r + self.params['c']*bonus_scalar, a, done)

    def policy(self, s, bonus_array):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acts)
        return self.maxAction(s, bonus_array)

    def maxAction(self, s, bonus_array):
        act_vals = self.Q[self.getIndex(s), :]
        move = argMax(act_vals + (1 - self.params['c'])*bonus_array)
        return move
