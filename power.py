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

class DensePowerDecomposition():
    def __init__(self, n_components=2, restore_zeros=False):
        self.n_components = n_components
        self.restore_zeros = restore_zeros

    def fit_transform(self, M):
        M = M.todense()
        zeros = np.where(M == 0.0)
        (nrows, ncols) = M.shape

        embedded = np.zeros((nrows, self.n_components))
        for i in range(self.n_components):
            b = np.random.rand(ncols)
            b /= np.linalg.norm(b)
            for iter in range(10):
                nextb = np.zeros(ncols)
                cols = list(range(ncols))
                random.shuffle(cols)
                for j in cols:
                    nextb[j] = M[j,:].dot(b)
                norm = np.linalg.norm(nextb)
                nextb /= norm
                b = nextb

            embedded[:,i] = b * norm

            c = np.mat(b)
            M -= norm * c.T * c

            # restore zeros to simulate sparse power decomposition
            if self.restore_zeros:
                M[zeros] = 0.0

        return embedded
 
class SparsePowerDecomposition():
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, M):
        M = M.tocsr()
        (nrows, ncols) = M.shape
        embedded = np.zeros((nrows, self.n_components))

        for cn in range(self.n_components):
            b = np.random.rand(ncols)
            b /= np.linalg.norm(b)
            for i in range(10):
                nextb = np.zeros(ncols)
                col_ids = list(range(ncols))
                random.shuffle(col_ids)
                for j in col_ids:
                    nextb[j] = M[j,:].dot(b)
                norm = np.max(np.abs(nextb))
                nextb /= norm
                b = nextb

            # Save it!
            embedded[:,cn] = b * norm

            b = b / np.linalg.norm(b)
            (rows, cols) = M.nonzero()
            for (k, (row, col)) in enumerate(zip(rows, cols)):
                M.data[k] -= norm * b[row] * b[col]
        return embedded


if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    algs = [
        ('dense-power', DensePowerDecomposition(10, False)),
        ('dense-power-sparsified', DensePowerDecomposition(10, True)),
        #('scipy-svd', ScipySvd(10)),
        #('sparse-power', SparsePowerDecomposition(10)),
    ]
    for (name, alg) in algs:
        print '\n\nResults for algorithm %s\n\n' % name
        embedding = alg.fit_transform(cosim.matrix)
        evaluator = Evaluator(cosim, embedding)
        evaluator.evaluate()

    #A = np.array([[1, 2, 0], [-2, 1, 2], [1, 3, 1]])
    #A = scipy.sparse.lil_matrix(A).tocsr()
    #pd = PowerDecomposition(1)
    #embedding = pd.fit_transform(A.T)
    #print A
