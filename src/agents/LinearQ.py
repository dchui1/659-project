import numpy as np
import math
from src.agents.Agent import Agent
from src.utils.math import argMax
from src.sparse import SparseTC

class LinearQ(Agent):
    def __init__(self, state_shape, num_acts, params):
        self.tiles = params['tiles']
        self.tilings = params['tilings']
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.alpha = params['alpha'] / float(self.tilings)

        self.state_shape = state_shape
        self.num_states = np.prod(state_shape)
        self.num_acts = num_acts

        self.tc = SparseTC({
            'tiles': self.tiles,
            'tilings': self.tilings,
            'dims': len(state_shape),
            'actions': num_acts,
        })

        self.size = self.tc.features()

        self.w = np.zeros((self.size))

    def maxAction(self, s):
        act_vals = [self.tc.representation(s, a).dot(self.w) for a in range(self.num_acts)]
        move = argMax(act_vals)
        return move

    def learn(self, s, sp, r, a, gamma):
        ap = self.maxAction(sp)

        Q_p = self.tc.representation(sp, ap).dot(self.w)

        x = self.tc.representation(s, a)
        Q = x.dot(self.w)
        tde = (r + gamma * Q_p) - Q
        self.w = self.w + self.alpha*tde*x

    def update(self, S, Sp, r, a, done):
        if done:
            self.learn(S, Sp, r, a, 0)
        else:
            self.learn(S, Sp, r, a, self.gamma)

    def print(self):
        print(self.w)
