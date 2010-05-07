#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np

import hadoopy

class Mapper(object):
    def __init__(self):
        self.out_sums = {}
        with open(os.environ["CLUSTERS_PKL"]) as fp:
            self.clusters, self.canopy_clusters = pickle.load(fp)
        self.nn = __import__(os.environ['NN_MODULE'],
                             fromlist=['nn']).nn
        
    def map(self, key, feat):
        feat = np.fromstring(feat, dtype=np.float32)
        canopies = key[1]
        cluster_ids = [self.canopy_clusters[x] for x in canopies if x in self.canopy_clusters]
        if not any(cluster_ids):
            return
        cluster_ids = list(set.union(*cluster_ids))
        clusters = self.clusters[cluster_ids]
        # Find NN using slow metric
        nearest_ind = cluster_ids[self.nn(feat, clusters)[0]]
        # Expand the array by 1 and use it to normalize later
        feat = np.resize(feat, feat.shape[0] + 1)
        feat[-1] = 1
        try:
            self.out_sums[nearest_ind] += feat
        except KeyError:
            self.out_sums[nearest_ind] = feat

    def close(self):
        for nearest_ind, feat in self.out_sums.iteritems():
            yield nearest_ind, feat.tostring()

def reducer(key, values):
    cur_cluster_sum = None
    for vec in values:
        vec = np.fromstring(vec, dtype=np.float32)
        try:
            cur_cluster_sum += vec
        except TypeError:
            cur_cluster_sum = vec
    center = cur_cluster_sum[0:-1] / cur_cluster_sum[-1]
    yield key, center.tostring()



if __name__ == "__main__":
    if hadoopy.run(Mapper, reducer):
        hadoopy.print_doc_quit(__doc__)
