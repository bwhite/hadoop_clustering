#!/usr/bin/env python
import os
import time
import cPickle as pickle

import numpy as np

import hadoopy

import profile

class Mapper(profile.ProfileJob):
    def __init__(self):
        super(Mapper, self).__init__()
        with open(os.environ["CLUSTERS_PKL"]) as fp:
            self.clusters = pickle.load(fp)
        out_sums_size = (len(self.clusters), len(self.clusters[0]) + 1)
        self.out_sums = np.zeros(out_sums_size, dtype=np.float32)
        self.out_sums_mask = np.ones(len(self.clusters), dtype=np.bool8)
        self.valid_clust = []
        self.nn = __import__(os.environ['NN_MODULE'],
                             fromlist=['nn']).nn

    def map(self, key, feat):
        # Extends the array by 1 dim that has a 1. in it
        feat = np.fromstring(feat + '\x00\x00\x80?', dtype=np.float32)
        nearest_ind = self.nn(feat[0:-1], self.clusters)[0]
        if self.out_sums_mask[nearest_ind]:
            self.out_sums_mask[nearest_ind] = 0
            self.valid_clust.append(nearest_ind)
        self.out_sums[nearest_ind] += feat

    def close(self):
        for nearest_ind, out_sum in zip(self.valid_clust, self.out_sums[self.valid_clust]):
            yield nearest_ind, out_sum.tostring()
        super(Mapper, self).close()

class Reducer(profile.ProfileJob):
    def __init__(self):
        super(Reducer, self).__init__()

    def reduce(self, key, values):
        cur_cluster_sum = None
        for vec in values:
            vec = np.fromstring(vec, dtype=np.float32)
            try:
                cur_cluster_sum += vec
            except TypeError:
                cur_cluster_sum = vec
        center = cur_cluster_sum[0:-1] / cur_cluster_sum[-1]
        yield key, center.tostring()
    
    def close(self):
        super(Reducer, self).close()


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
