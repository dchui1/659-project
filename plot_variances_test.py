import numpy as np
import matplotlib.pyplot as plt


data_dict1 = np.load('tmp/TabR_TabQ_maxvars.npy')
max_var_T = data_dict1.item().get('max_var_T_array')
max_bonuses = data_dict1.item().get('max_bonus')

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(max_var_T)
# plt.title("Maximum T-Distribution Variances Across All Episodes")
# plt.xlabel("Timesteps")
# plt.ylabel("Maximumum Variance over all (s, a) pairs")
# plt.show()

data_dict2 = np.load('tmp/TabR_TabQ_trajectory.npy')
all_variances = data_dict2.item().get('t_dist_var_along_trajectory')
bonuses = data_dict2.item().get('all_bonuses')
all_vars = np.array(all_variances)
size = all_vars.shape
num_acts = size[0]
num_states = size[1]


mins = []
maxs = []
booleans2 = []
for j in range(num_acts):
    min_a = []
    max_a = []
    bool_a = []
    for i in range(num_states):
        min_s = []
        min_a.append(min_s)
        max_s = []
        max_a.append(max_s)
        bool_s = []
        bool_a.append(bool_s)
    mins.append(min_a)
    maxs.append(max_a)
    booleans2.append(bool_a)


# function to test if a list is in decreasing order
def ordertest(A):
    for i in range( len(A) - 1 ):
        if A[i] < A[i+1]:
            return False
    return True


booleans = []
for a in range(num_acts):
    for s in range(num_states):
        v = all_vars[a, s]
        truth_val = ordertest(v)
        booleans.append(truth_val)
        mins[a][s] = min(all_vars[a, s])
        maxs[a][s] = max(all_vars[a, s])
        booleans2[a][s] = truth_val

mins = np.array(mins)
maxs = np.array(maxs)
booleans2 = np.array(booleans2)
