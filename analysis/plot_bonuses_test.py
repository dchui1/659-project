import numpy as np
import matplotlib.pyplot as plt


data_dict = np.load('tmp/TabR_TabQ_trajectory.npy')
all_bonuses = data_dict.item().get('all_bonuses')
vars = data_dict.item().get('t_dist_var_along_trajectory')

b = np.array(all_bonuses)
v = np.array(vars)
size_b = b.shape
size_v = v.shape
print(size_b)
print(size_v)
assert size_b == size_v


# data_dict1 = np.load('tmp/TabR_TabQ_maxvars.npy')
# max_var_T = data_dict1.item().get('max_var_T_array')
# max_bonuses = data_dict1.item().get('max_bonus')
# data_dict2 = np.load('tmp/TabR_TabQ_trajectory.npy')
# all_variances = data_dict2.item().get('t_dist_var_along_trajectory')
# bonuses = data_dict2.item().get('all_bonuses')
# all_vars = np.array(all_variances)

num_acts = size_b[0]
num_states = size_b[1]


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


# booleans = []
# for a in range(num_acts):
#     for s in range(num_states):
#         b_sa_list = b[a, s]
#         truth_val = ordertest(b_sa_list)
#         booleans.append(truth_val)
#         mins[a][s] = min(b[a, s])
#         maxs[a][s] = max(b[a, s])
#         booleans2[a][s] = truth_val
#
# mins = np.array(mins)
# maxs = np.array(maxs)
# booleans2 = np.array(booleans2)

# only for s = (0, 0) and all actions
booleans = []
for a in range(num_acts):
# a = 1
    b_sa_list = b[a, 0]
    truth_val = ordertest(b_sa_list)
    booleans.append(truth_val)
    # mins[a][0] = min(b[a, 0])
    # maxs[a][0] = max(b[a, 0])
    booleans2[a][0] = truth_val

mins = np.array(mins)
maxs = np.array(maxs)
booleans2 = np.array(booleans2)


# for a in range(num_acts):
#     fig = plt.figure()
#     ax = plt.axes()
#     ax.plot(b[a, 0])
#     plt.title("Bonuses for action, s = (0, 0) across all Timesteps")
#     plt.xlabel("Timesteps")
#     plt.ylabel("B( (0, 0), action ) values")
#     plt.show()
