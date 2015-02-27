import random
import sys
import numpy as np
import scipy.sparse
import scipy.stats
from sklearn.neighbors import NearestNeighbors

from utils import *

SAMPLE_SIZE = 100
TOP_K = 5


class Evaluator:
    def __init__(self, cosim, embedding, out=sys.stdout):
        self.cosim = cosim
        self.out = out
        self.sample_size = SAMPLE_SIZE
        self.examples = EXAMPLE_ARTICLES
        self.topk = TOP_K
        self.embedding = to_dense(embedding)
        self.nbrs = NearestNeighbors()
        self.nbrs.fit(self.embedding)

    def evaluate(self):
        self.evaluate_examples()
        self.evaluate_precision()
        self.evaluate_correlation()

    def evaluate_examples(self):
        for title in self.examples:
            id1 = self.cosim.get_article_id(title)
            self.write('\nneighbors for %s (id=%d)\n' % (title, id1))
            actual = self.actual_neighbors(id1, self.topk)
            embedded = self.embedded_neighbors(id1, self.topk)
            self.write('    %-35s   %-35s\n' % ('--Actual--', '--Embedded--'))
            for (a_id, a_sim), (e_id, e_sim) in zip(actual, embedded):
                self.write('    %.3f %-25s   %.3f %-25s\n' % (
                        a_sim, self.cosim.titles[a_id],
                        e_sim, self.cosim.titles[e_id]))

    def write(self, message):
        self.out.write(message)

    def actual_neighbors(self, id1, n):
        v_actual = self.cosim.matrix[id1].toarray()[0]
        return [ (id2, self.cosim.matrix[id1,id2])
                 for id2 in np.argsort(-v_actual)[:n] ]

    def embedded_neighbors(self, id1, n):
        (scores, indices) = self.nbrs.kneighbors(self.embedding[id1], n)
        return zip(indices[0], np.exp(-scores[0]))

    def evaluate_precision(self):
        ids = self.sample_ids()
        self.write('\nPrecision of embedded neighbors compared to actual neighbors\n')
        for k in (1, 5, 20, 50, 100):
            hits = 0
            for id1 in ids:
                actual = set(id2 for id2, score in self.actual_neighbors(id1, k))
                embedded = set(id2 for id2, score in self.embedded_neighbors(id1, k))
                hits += len(actual.intersection(embedded))
            precision = 100.0 * hits / (k * len(ids))
            self.write('\tprecision of nearest-%d: precision %.3f%%\n' % (k, precision))

    def evaluate_correlation(self, num_neighbors=3, num_zeros=1):
        X = []
        Y = []
        for id1 in self.sample_ids():
            indices = self.cosim.matrix[id1].indices
            for id2 in random.sample(indices, 3):
                X.append(np.linalg.norm(self.embedding[id1] - self.embedding[id2]))
                Y.append(self.cosim.matrix[id1,id2])
        self.write('\nCorrelations between actual and embedded distances\n')
        self.write('\tPearsons: %.3f\n' % scipy.stats.pearsonr(X, Y)[0])
        self.write('\tSpearmans: %.3f\n' % scipy.stats.spearmanr(X, Y)[0])

    def sample_ids(self):
        return random.sample(self.cosim.titles, self.sample_size)
        
def to_dense(M):
    if scipy.sparse.issparse(M):
        return M.toarray()
    else:
        return M
        
if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    evaluator = Evaluator(cosim, cosim.matrix)
    evaluator.evaluate()
