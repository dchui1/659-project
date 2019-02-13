import numpy as np

class TileCoding:
    def __init__(self, dim, num_tiling, num_tile, num_action=1):
        self.dim = dim
        self.num_tiling = num_tiling
        self.num_tile = num_tile
        self.tile_one_tiling = self.num_tile ** self.dim
        self.num_action = num_action
        self.len_tile = 1.0 if self.num_tile == 1 else 1.0 / (self.num_tile - 1)
        self.tiling_dist = 0.0 if self.num_tiling == 1 else self.len_tile / (self.num_tiling - 1)

    def get_index(self, pos, action=None):
        assert len(pos) == self.dim
        index = np.zeros((self.num_tiling))
        for ntl in range(self.num_tiling):
            ind = 1
            for d in range(self.dim):
                if pos[d] != 1.0:
                    ind += (pos[d] - self.tiling_dist * ntl) // self.len_tile * self.num_tile**d
                else:
                    ind += (pos[d] - self.tiling_dist * ntl) // self.len_tile * self.num_tile**d - 1
            index[ntl] = ind + self.tile_one_tiling * ntl
        if action != None:
            index += action * self.tile_one_tiling * self.num_tiling
        return index.astype(int)