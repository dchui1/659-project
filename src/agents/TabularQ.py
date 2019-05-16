import numpy as np
from src.agents.Agent import Agent
from src.utils.math import argMax

class TabularQ(Agent):
    def __init__(self, state_shape, num_acts, params):
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.state_shape = state_shape
        self.num_states = np.prod(state_shape)
        self.num_acts = num_acts

        self.Q = np.zeros((self.num_states, self.num_acts))

    def getIndex(self, s):
        return np.ravel_multi_index(s, self.state_shape)

    def policy(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acts)
        return self.maxAction(s)

    def maxAction(self, s):
        act_vals = self.Q[self.getIndex(s), :]
        move = argMax(act_vals)
        return move

    def learn(self, s, sp, r, a, gamma):
        ap = self.maxAction(sp)
        Q_p = self.Q[self.getIndex(sp), ap]
        s_idx = self.getIndex(s)

        tde = (r + gamma * Q_p) - self.Q[s_idx, a]
        self.Q[s_idx, a] = self.Q[s_idx, a] + self.alpha*tde

    def update(self, S, Sp, r, a, done):
        gamma = 0 if done else self.gamma
        self.learn(S, Sp, r, a, gamma)

    def print(self):
        print(self.Q)
