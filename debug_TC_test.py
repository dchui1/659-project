'''
The TC will give unique feature vectors as long as `tiles` = number of elements in each dimension.
i.e., if we have x_coord = 0.0, 0.5, 1.0, we want 3 tiles.
'''

import numpy as np
import itertools
from utils.sparse import SparseTC


def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)


num_acts = 4
x_min = 0.0
y_min = 0.0
alpha = 0.5
x_max = 1.0
y_max = 1.0
x_array = np.array(np.linspace(x_min, x_max, 3))
y_array = x_array

M = []
for row_i, x_i in enumerate(x_array):
    xy_list = list(itertools.product([x_i], y_array))
    M.append(xy_list)
M = np.array(M)
M = np.concatenate(M, 0)

tc = SparseTC({
    'tiles': 3,
    'tilings': 1,
    'dims': len([1, 1]),
    'actions': num_acts,
})

F = []
for a in range(num_acts):
    for coordinate in M:
        x = tc.representation(coordinate, a).array()
        # x_mat = x.reshape(1, len(x))
        F.append(x)

one_idx = []
for i, arr in enumerate(F):
    for entry, j in enumerate(arr):
        if entry == 1.0:
            one_idx.append((i, j))

print(allUnique(one_idx))
