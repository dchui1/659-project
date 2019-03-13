import numpy as np
from agents.TabularQ import TabularQ
from bayesianapproximator import TabularBayesianApproximation
from utils.sparse import SparseTC


class TabularRTabularQ(TabularQ):
    def __init__(self, state_shape, num_acts, params):
        super().__init__(state_shape, num_acts, params)
        self.rewardApprox = TabularBayesianApproximation(state_shape, num_acts)

        self.data = []
        self.reward_data = []
        self.state_shape = state_shape
        # self.tc = SparseTC({
        #     'tiles': 2,
        #     'tilings': 1,
        #     'dims': len(state_shape),
        #     'actions': num_acts,
        # })

    def update(self, s, sp, r, a, done):
        x = self.getIndex(s) + (a * self.num_states)

        # create data to be taken by the Tile Coder and stored
        # x_coord = s[0]/self.state_shape[0]
        # y_coord = s[1]/self.state_shape[1]
        # s_normalized = (x_coord, y_coord)
        # fx = self.tc.representation(s_normalized, a)\
        #     .array()
        # x_mat = fx.reshape(1, len(fx))
        # r_mat = np.array([[r]])
        # self.data.append(x_mat)
        # self.reward_data.append(r_mat)

        self.rewardApprox.update_stats(x, r)
        samples = self.rewardApprox.sample(x, 50)
        bonus = samples[0]
        super().update(s, sp, r + bonus, a, done)
