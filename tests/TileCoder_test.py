# test the TileCoder
from utils.TileCoding import TileCoding
from utils.sparse import SparseTC
import numpy as np
# action 0 = RIGHT
# action 1 = UP
# action 2 = LEFT
# action 3 = DOWN


def debug_tc(x, y, a):
    s = np.array([x, y])
    sptc = SparseTC({
        'tiles': 2,
        'tilings': 1,
        'dims': len([1, 1]),
        'actions': 4,
    })
    x = sptc.representation(s, a).array()
    idx = sptc.tc.get_index(s, a)
    return x, idx


dim = 2
num_tiling = 1
num_tile = 2
tile_one_tiling = num_tile ** dim
num_action = 4
len_tile = 1.0 if num_tile == 1 else 1.0 / (num_tile - 1)
tiling_dist = 0.0 if num_tiling == 1 else len_tile / (num_tiling - 1)


def get_index(pos, action=None):
    assert len(pos) == dim
    index = np.zeros(num_tiling)
    for ntl in range(num_tiling):
        ind = 0
        for d in range(dim):
            ind += ((pos[d] - tiling_dist * ntl) + 1e-12) // len_tile * num_tile**d
        index[ntl] = ind + tile_one_tiling * ntl
    if action != None:
        index += action * tile_one_tiling * num_tiling
    return index.astype(int)
