import numpy as np
import math
from agents.Agent import Agent
from utils.math import argMax
from utils.TileCoding import TileCoding

class LinearQ(Agent):
    def __init__(self, state_shape, num_acts):
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.tiles = 30
        self.tilings = 1

        self.state_shape = state_shape
        self.num_states = np.prod(state_shape)
        self.num_acts = num_acts
        self.size = int(math.pow(self.tiles, len(state_shape)) * self.tilings * self.num_acts)

        self.t = TileCoding(len(self.state_shape), self.tilings, self.tiles, self.num_acts)

        self.w = np.zeros((self.size))
        self.next_action = 0

    def getIndex(self, s):
        return np.ravel_multi_index(s, self.state_shape)

    def policy(self, S):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acts)
        return self.maxAction(S)

    def maxAction(self, s):
        act_vals = [self.w[self.t.get_index(s/29, a)[0]] for a in range(self.num_acts)]
        move = argMax(act_vals)
        return move

    def getAction(self, Obs):
        return self.next_action

    def start(self, obs):
        self.next_action = self.policy(obs)
        return self.next_action

    # if gamma_tp1 = 0, that means the episode terminated
    def learn(self, s, sp, r, a, gamma):
        ap = self.maxAction(sp)

        Q_p = self.w[self.t.get_index(sp/29, ap)[0]]

        s_idx = self.t.get_index(s/29, a)[0]
        x = np.zeros((self.size))
        x[s_idx] = 1

        tde = (r + gamma * Q_p) - self.w[s_idx]  # add a max_bonus i
        self.w = self.w + self.alpha*tde*x

    def update(self, S, Sp, r, a, done):
        if done:
            self.learn(S, Sp, r, a, 0)
        else:
            self.next_action = self.policy(Sp)
            self.learn(S, Sp, r, a, self.gamma)

    def print(self):
        print(self.w.reshape((900, 4)))
