import numpy as np
from src.agents.TabularQ import TabularQ
from src.agents.Agent import Agent
from src.utils.math import argMax

class Dual_Vf(Agent):
    def __init__(self, state_shape, num_acts, params):
      super().__init__(state_shape, num_acts, params)
      self.epsilon = params["epsilon"]
      self.TabR = TabularQ(state_shape, num_acts, params["TabR"])
      self.TabB = TabularQ(state_shape, num_acts, params["TabB"])
      self.num_states = np.prod(state_shape)

    def update(self, s, sp, r, bonus_scalar, a, done):
        self.TabR.update(s, sp, r, a, done)
        self.TabB.update(s, sp, bonus_scalar, a, done)

    def getIndex(self, s):
        return np.ravel_multi_index(s, self.state_shape)

    def policy(self, s, bonus_array):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acts)

        act_vals = self.params["c"]*self.TabR.Q[self.getIndex(s), :] + (1 - self.params["c"])*self.TabB.Q[self.getIndex(s), :]
        print(act_vals)
        move = argMax(act_vals)
        return move
