import lib.python.sparse as sparse
from utils.TileCoding import TileCoding
import numpy as np

# This is a sparse vector for doing very fast dot products and additions.
# It is approximately 10x faster than scipy.sparse but it is not nearly as complete.
# Right now, it has not been tested for matrix/vector multiplications
class SparseOneVector(object):
    def __new__(cls, ones, length, _T = False):
        m = object.__new__(cls)
        m.ones = ones
        if not _T:
            m.shape = (length,)
            m.T = cls(ones, length, _T = True)
        else:
            m.shape = (length, 1)
        return m

    # Get back a numpy array (in case you need a feature that isn't/can't be implemented here)
    def array(self):
        m = np.zeros(self.shape)
        m[self.ones] = 1
        return m

    # Don't use me. Use dot instead
    def outer(self, V):
        out = np.zeros((self.shape[0], V.shape[1]))
        out[self.ones, :] = V.copy()
        return out

    # Don't use me. Use dot instead
    def vdot(self, V):
        return sparse.VectorDotProduct(self.ones, V)

    # Don't use me. Use dot instead
    def mdot(self, V):
        return sparse.MatrixDotProduct(self.ones, V)

    # Don't use me. Use dot instead
    def vmProd(self, V):
        m = sparse.VectorMatrixProduct(list(self.ones), V)
        # out = np.zeros((self.shape[0], V.shape[1]))
        # for c in range(V.shape[1]):
        #     out[0, c] = float(sum(V[self.ones, c]))

        return m

    # Should operate exactly as numpy's dot method.
    def dot(self, V):
        if len(V.shape) == 2:
            out = []
            if self.shape[1] == 1 and V.shape[0] == 1: # Outer product
                out = self.outer(V)
            elif self.shape[0] == 1 and V.shape[0] == 1: # must be a vector dot product
                out = self.vdot(V)
            elif self.shape[0] == 1 and V.shape[1] == 1: # must be a matrix dot product
                out = self.mdot(V)
            elif self.shape[0] == 1 and V.shape[1] > 1: # vector-matrix product
                out = self.vmProd(V)
            else:
                raise IndexError
            return out
        else:
            return np.array(float(sum(V[self.ones])))

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if len(other.shape) == 2:
            raise NotImplementedError
        m = other.copy()
        m[self.ones] = 1 + m[self.ones]
        return m

    def __sub__(self, other):
        if len(other.shape) == 2:
            return self.array() - other
        else:
            return self.array() - other

    def __rsub__(self, other):
        if len(other.shape) == 2:
            m = other.copy()
            m[0, self.ones] = m[0, self.ones] - 1
            return m
        else:
            m = other.copy()
            m[self.ones] = m[self.ones] - 1
            return m

    def __mul__(self, other):
        m = np.zeros(self.shape)
        m[self.ones] = other
        return m

    def __rmul__(self, other):
        return self.__mul__(other)

class SparseTC(object):
    def __init__(self, args):
        self.tilings = args['tilings']
        self.tiles = args['tiles']
        self.dims = args['dims']
        self.actions = args['actions']

        self.tc = TileCoding(self.dims, self.tilings, self.tiles, self.actions)

    def features(self):
        return int(self.tiles**self.dims * self.tilings * self.actions)

    def representation(self, s, a):
        idx = self.tc.get_index(s, a)
        vec = SparseOneVector(list(idx), self.features())
        return vec

