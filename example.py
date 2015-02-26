from utils import *

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import KernelCenterer

cosim = Cosimilarity('dat/1000-export')
cosim.normalize()

km = KMeans(n_clusters=12)
km.fit(cosim.matrix)
for (i, centroid) in enumerate(km.cluster_centers_):
    top_ids = np.argsort(-centroid)[:5]
    print 'top titles in centroid', i
    for id in top_ids:
        print '\t%s:%.3f' % (cosim.titles[id], centroid[id])
