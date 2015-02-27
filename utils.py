import numpy as np
import scipy.sparse
import sklearn.preprocessing
import sys

EXAMPLE_ARTICLES = [
    'Firefly (TV series)',
    #'Albatross',
    #'2010 Winter Olympics',
    #'Jessica Alba',
    'Christianity',
    'Computer',
]

def warn(message):
    sys.stderr.write(message + '\n')

class Cosimilarity:
    def __init__(self, path):
        warn('loading matrix from %s' % `path`)
        self.titles = self.read_titles(path + '/ids.txt')
        warn('read %d titles' % len(self.titles))

        self.matrix = self.read_matrix(path + '/matrix.txt')
        warn('read %d by %d matrix with %s non-zero entries' %
                (    self.matrix.shape[0], 
                     self.matrix.shape[1],
                     len(self.matrix.nonzero()[0])  ))
        self.matrix = self.matrix.tocsr()

    def read_titles(self, path):
        titles = {}
        for line in open(path):
            (dense_id, sparse_id, title) = line.split('\t')
            titles[int(dense_id)] = title.strip()
        return titles

    def read_matrix(self, path):
        n = max(self.titles.keys()) + 1
        data = []
        rows = []
        cols = []
    
        ids = []
        for (i, line) in enumerate(open(path), 1):
            if i not in self.titles:
                continue
            ids.append(i)
            if i % 1000 == 0:
                print 'reading line', i
            for token in line.split():
                j, sim = token.split(':')
                j = int(j)
                if j in self.titles:
                    data.append(float(sim))
                    rows.append(i)
                    cols.append(j)
    
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))

    def normalize(self):
        M = self.matrix.toarray()
        sklearn.preprocessing.normalize(M, axis=0, copy=False)
        sklearn.preprocessing.normalize(M, axis=1, copy=False)
        M = 0.5 * M + 0.5 * M.T
        self.matrix = scipy.sparse.csr_matrix(M)
            

    def get_article_id(self, title):
        for (i, t) in self.titles.items():
            if clean_title(title) == clean_title(t):
                return i
        return None

    def print_neighbors(self, M=None, n=5, articles=EXAMPLE_ARTICLES, out=sys.stdout):
        if not M: M = self.matrix
        for a in articles:
            id = cosim.get_article_id(a)
            v = M[id].toarray()[0]
            out.write('neighbors for %s (id=%d)\n' % (a, id))
            top_ids = np.argsort(-v)[:n]
            for id in top_ids:
                out.write('\t%s:%.3f\n' % (self.titles[id], v[id]))

def clean_title(t):
    return t.lower().replace('_', ' ').replace('+', ' ')

if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
    cosim.print_neighbors()
