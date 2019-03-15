import sparse
import numpy as np

def vdot(s, V):
    return np.array(float(sum(V[0, s])))

def mdot(s, V):
    return np.array(float(sum(V[s, 0])))

def vmProd(s, V):
    out = np.zeros((1, V.shape[1]))
    for c in range(V.shape[1]):
        out[0, c] = float(sum(V[s, c]))
    return out


for run in range(100):
    # test vmProd
    n = np.random.randn(512, 512)
    s = [i for i in range(0, 512, 20)]
    m = sparse.VectorMatrixProduct(s, n)
    o = vmProd(s, n)
    print(np.linalg.norm(o - m))

    # test mdot
    n = np.random.rand(512, 1)
    m = sparse.MatrixDotProduct(s, n)
    o = mdot(s, n)
    print(np.linalg.norm(o - m))

    # test vdot
    n = np.random.randn(1, 512)
    m = sparse.VectorDotProduct(s, n)
    o = vdot(s, n)
    print(np.linalg.norm(o - m))
