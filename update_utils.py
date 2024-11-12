import numpy as np


def proj_tangent(x, differential):
    d = differential.shape[1]
    r = differential.shape[0]
    xd = x.T@differential
    return differential - 0.5* x@(xd + xd.T)
    
def data_gen_pca(N, n, m):
    A = np.zeros((N, n, m))
    for i in range(N):
        A[i] = np.random.randn(n,m)
    return A