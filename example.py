from utils import *

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import KernelCenterer

cosim = Cosimilarity('dat/1000-export')

# Recenter matrix. This converts it to a dense matrix :(
kc = KernelCenterer()
dense = kc.fit_transform(cosim.matrix.toarray())

km = KMeans(n_clusters=12)
km.fit(dense)
for (i, centroid) in enumerate(km.cluster_centers_):
    top_ids = np.argsort(-centroid)[:5]
    print 'top titles in centroid', i
    for id in top_ids:
        print '\t%s:%.3f' % (cosim.titles[id], centroid[id])
