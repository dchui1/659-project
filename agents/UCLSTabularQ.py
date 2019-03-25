import numpy as np

from agents.TabularQ import TabularQ
from utils.ucls_new import UCLS

class UCLSTabularQ(TabularQ):

    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.ucls = UCLS(state_shape, num_acts, params)


    # def update(self, S, Sp, reward, a, done):
    #
    #     if not done:
    #         next_action = self.policy(Sp)
    #         self.populate_td_features(Sp, next_action)
    #     else:
    #         self.populate_td_features(state=None)
    #
    #     td_error = reward - self.get_value()
    #     self.step_all(reward, td_error)
    #
    #     if not done:
    #         self.current_state[:] = state
    #         self.current_action = next_action
    #         return self.current_action

    def policy(self, S):
        self.ucls.policy(S)

    def learn(self, s, sp, r, a, gamma):
        self.ucls.populate_td_features(sp, a)

        td_error = reward - self.get_value()
        self.ucls.step_all(reward, td_error)

        # if not done:
        #     self.current_state[:] = state
        #     self.current_action = next_action
        #     return self.current_action
