from src.RLGlue.BaseAgent import BaseAgent
import numpy as np
from src.bayesian_inference.BNNApproximation import BNNApproximation

# Takes an agent and converts its API to one that works with RLGlue
# keeps track of the previous (state, action) pair
# and passes that to the agent update function
class LinearBayesianBonusGenerator(BaseAgent):
    def __init__(self, base_agent, params):
        self.agent = base_agent
        self.rewardApprox = BNNApproximation(self.agent.state_shape, self.agent.num_acts, params)
        self.feature_shape = np.prod(self.agent.state_shape) * self.agent.num_acts
        self.s_t = None
        self.a_t = None

    # Called at the beginning of each episode
    # Takes the initial state from environment
    def start(self, s):
        self.s_t = s
        bonus_array = self.compute_bonus_array(s)
        self.a_t = self.agent.policy(s, bonus_array)
        return self.a_t

    # Called on each timestep (after the first)
    # updates the agent's value function and gets the next action
    def step(self, r_t, s_tp1):
        bonus_array = self.compute_bonus_array(self.s_t)
        bonus_t = bonus_array[self.a_t]
        self.agent.update(self.s_t, s_tp1, r_t, bonus_t, self.a_t, False)

        x = self.get_onehot(self.s_t, self.a_t)
        self.rewardApprox.update_stats(x, r_t)

        self.s_t = s_tp1
        self.a_t = self.agent.policy(s_tp1, self.compute_bonus_array(s_tp1))

        return self.a_t

    # Called whenever the agent transitions into a terminal state
    # There is no next state, so just pass a dummy to the agent
    def end(self, r):
        bonus_array = self.compute_bonus_array(self.s_t)
        bonus_t = bonus_array[self.a_t]

        self.agent.update(self.s_t, self.s_t, r, bonus_t, self.a_t, True)

        x = self.get_onehot(self.s_t, self.a_t)
        self.rewardApprox.update_stats(x, r)

    def compute_bonus_array(self, s):
        bonus_array = []
        for a in range(self.agent.num_acts):
            o = self.get_onehot(s, a)
            bonus = self.rewardApprox.sample(o, n=100)
            bonus_array.append(bonus)

        return np.array(bonus_array)

    def getIndex(self, s):
        return np.ravel_multi_index(s, self.agent.state_shape)

    def get_onehot(self, s, a):
        x = self.getIndex(s) + (a * self.agent.num_states)
        o = np.zeros(self.feature_shape)
        o[x] = 1.0
        return o
