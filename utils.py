import scipy.sparse
import sys


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

if __name__ == '__main__':
    cosim = Cosimilarity('./dat/1000-export')
