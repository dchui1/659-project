import numpy as np
from agents.Agent import Agent
from utils.math import argMax

class TabularQ(Agent):
    def __init__(self, state_shape, num_acts, params):
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.state_shape = state_shape
        self.num_states = np.prod(state_shape)
        self.num_acts = num_acts

        self.Q = np.zeros((self.num_states, self.num_acts))
        self.next_action = 0

    def getIndex(self, s):
        return np.ravel_multi_index(s, self.state_shape)

    def policy(self, S):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_acts)
        return self.maxAction(S)

    def maxAction(self, s):
        act_vals = self.Q[self.getIndex(s), :]
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
        Q_p = self.Q[self.getIndex(sp), ap]

        s_idx = self.getIndex(s)

        tde = (r + gamma * Q_p) - self.Q[s_idx, a]  # add a max_bonus i
        self.Q[s_idx, a] = self.Q[s_idx, a] + self.alpha*tde

    def update(self, S, Sp, r, a, done):
        if done:
            self.learn(S, Sp, r, a, 0)
        else:
            self.next_action = self.policy(Sp)
            self.learn(S, Sp, r, a, self.gamma)

    def print(self):
        print(self.Q)
