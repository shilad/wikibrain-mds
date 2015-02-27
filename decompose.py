# Experiments with existing decomposition algorithms


from utils import *
from evaluate import Evaluator

import sklearn.decomposition

#ALGS = [ 'PCA', 'KernelPCA', 'RandomizedPCA', 'NMF', 'TruncatedSVD' ]
ALGS = [ 'RandomizedPCA', 'TruncatedSVD' , 'NMF' ]
NUM_COMPONENTS = [2, 5, 10, 20 , 50, 100, 200]


def main(cosim):
    #dense = cosim.matrix.toarray()
    for nc in NUM_COMPONENTS:
        print
        print
        print '='*80
        print 'Results for all algorithms with %d components' % nc
        print '='*80
        print

        #evaluator.evaluate()
        for name in ALGS:
            f = getattr(sklearn.decomposition, name)
            alg = f(n_components=nc)
            embedding = alg.fit_transform(cosim.matrix)
            evaluator = Evaluator(cosim, embedding)

            print
            print 'results for', name, 'rank', nc, ':'
            evaluator.evaluate_examples()
            evaluator.evaluate_precision()
            evaluator.evaluate_correlation()
    

if __name__ == '__main__':
    cosim = Cosimilarity('./dat/5000-export')
    main(cosim)
