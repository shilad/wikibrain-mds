# Experiments with existing decomposition algorithms

import numpy as np


from utils import *
from evaluate import Evaluator

import sklearn.manifold as manifold
from sklearn.metrics.pairwise import euclidean_distances

NUM_COMPONENTS = [10, 20 , 50, 100, 200]

def mds(disparities, n_components, max_iter=300):
    n_samples = disparities.shape[0]

    disp_flat = ((1 - np.tri(n_samples)) * disparities).ravel()
    disp_flat_w = disp_flat[disp_flat != 0]

    X = np.random.rand(n_samples * n_components)
    X = X.reshape((n_samples, n_components))

    old_stress = None
    for it in range(max_iter):
        dis = euclidean_distances(X)

        # Compute stress (e.g. error)
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        old_stress = stress / dis

    return X


def main(cosim):
    for nc in NUM_COMPONENTS:
            dense = cosim.matrix.todense()
            affinity = 0.5 * dense + 0.5 * dense.T
            distance = np.maximum(1.0 - affinity, 0)
            #embedding = manifold.mds._smacof_single(D, n_components=nc)[0]
            embedding = mds(D, n_components=nc)

            print 'results for %d components\n\n\n' % nc
            evaluator = Evaluator(cosim, embedding)
            evaluator.evaluate()
    

if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    main(cosim)
    #cosim = Cosimilarity('./dat/5000-export')
    #main(cosim, SPARSE_ALGS)
