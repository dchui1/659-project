import numpy as np

from agents.TabularQ import TabularQ
from utils.ucls_new import UCLS

class UCLSTabularQ(TabularQ):

    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.ucls = UCLS(state_shape, num_acts, params)


    def policy(self, S):

        return self.ucls.policy(S)

    def learn(self, s, sp, reward, a, gamma):


        self.ucls.populate_td_features(sp, a)

        td_error = reward - self.ucls.get_value()
        self.ucls.step_all(reward, td_error)

        # if not done:
        #     self.current_state[:] = state
        #     self.current_action = next_action
        #     return self.current_action
