import numpy as np
import random

import scipy.sparse

from utils import *
from evaluate import Evaluator


class ScipySvd():
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, M):
        (w, v) = scipy.sparse.linalg.eigs(M, self.n_components)
        return v * w

class LanczosDecomposition():
    def __init__(self, n_components=2, restore_zeros=False):
        self.n_components = n_components
        self.restore_zeros = restore_zeros

    def fit_transform(self, M):
        (nrows, ncols) = M.shape

        k = 3 * self.n_components / 2

        Q = np.zeros((k+2, ncols))
        Q[0] = np.zeros(ncols)  # redundant, but added for clarity
        Q[1] = np.random.rand(ncols)
        Q[1] /= np.linalg.norm(Q[1])

        A = np.zeros(k+1)
        B = np.zeros(k+2)
        B[0] = 0.0  # redundant, but for clarity

        for i in range(1, k+1):
            w = M.dot(Q[i]) - B[i] * Q[i-1]
            A[i] = w.dot(Q[i])
            w -= A[i] * Q[i]

            # re-orthoganilzation
            D = np.zeros(w.shape)
            for j in range(1, i):
                D += w.dot(Q[j]) * Q[j]
            w -= D

            B[i+1] = np.linalg.norm(w)
            if B[i+1] == 0:
                break
            Q[i+1] = w / B[i+1]

        data = [B[2:k+1], A[1:k+1], B[2:k+1]]
        diag = [-1, 0, 1]

        # Calculate the decomposition
        triD = scipy.sparse.diags(data, diag, (k, k), format='csr')
        (w, v) = np.linalg.eig(triD.todense())

        return np.dot(Q[1:k+1,:].T, np.multiply(v, w))

class CompactLanczosDecomposition():
    def __init__(self, n_components=2, restore_zeros=False):
        self.n_components = n_components
        self.restore_zeros = restore_zeros

    def fit_transform(self, M):
        (nrows, ncols) = M.shape

        k = 3 * self.n_components / 2

        Q = np.zeros((k+2, ncols))
        Q[0] = np.zeros(ncols)  # redundant, but added for clarity
        Q[1] = np.random.rand(ncols)
        Q[1] /= np.linalg.norm(Q[1])

        A = np.zeros(k+1)
        B = np.zeros(k+2)
        B[0] = 0.0  # redundant, but for clarity

        for i in range(1, k+1):
            w = M.dot(Q[i]) - B[i] * Q[i-1]
            A[i] = w.dot(Q[i])
            w -= A[i] * Q[i]

            # re-orthoganilzation
            D = np.zeros(w.shape)
            for j in range(1, i):
                D += w.dot(Q[j]) * Q[j]
            w -= D

            B[i+1] = np.linalg.norm(w)
            if B[i+1] == 0:
                break
            Q[i+1] = w / B[i+1]

        data = [B[2:k+1], A[1:k+1], B[2:k+1]]
        diag = [-1, 0, 1]

        # Calculate the decomposition
        triD = scipy.sparse.diags(data, diag, (k, k), format='csr')
        (w, v) = np.linalg.eig(triD.todense())

        return np.dot(Q[1:k+1,:].T, np.multiply(v, w))
 
if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    algs = [
        ('lanczos', LanczosDecomposition(20, False)),
        ('scipy-svd', ScipySvd(20)),
    ]
    for (name, alg) in algs:
        print '\n\nResults for algorithm %s\n\n' % name
        embedding = alg.fit_transform(cosim.matrix)
        evaluator = Evaluator(cosim, embedding)
        evaluator.evaluate()
