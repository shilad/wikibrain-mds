# Experiments with existing decomposition algorithms


from utils import *
from evaluate import Evaluator

import sklearn.manifold as manifold

NUM_COMPONENTS = [10, 20 , 50, 100, 200]


def main(cosim):
    dense = cosim.matrix.todense()
    affinity = 0.5 * dense + 0.5 * dense.T
    distance = np.maximum(1.0 - affinity, 0)
    for nc in NUM_COMPONENTS:
        algs = [
            ('isomap', manifold.Isomap(nc)),
            #('TSNE', manifold.TSNE(nc, metric='precomputed')),
            ('spectral', manifold.SpectralEmbedding(nc, affinity='precomputed')),
            ('MDS', manifold.MDS(nc, dissimilarity='precomputed')),
        ]
        print
        print
        print '='*80
        print 'Results for all algorithms with %d components' % nc
        print '='*80
        print

        for name, alg in algs:
            M = distance
            if name in ('spectral', ): M = affinity
            embedding = alg.fit_transform(M)
            evaluator = Evaluator(cosim, embedding)

            print
            print 'results for', name, 'rank', nc, ':'
            evaluator.evaluate()
    

if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    main(cosim)
    #cosim = Cosimilarity('./dat/5000-export')
    #main(cosim, SPARSE_ALGS)
