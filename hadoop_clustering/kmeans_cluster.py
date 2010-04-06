#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np

import hadoopy
from vitrieve_algorithms import knearest_neighbor_l1
#from vitrieve_algorithms import nearest_neighbor


class Mapper(object):
    def __init__(self):
        self.out_sums = {}
        with open(os.environ["CLUSTERS_PKL"]) as fp:
            self.clusters = np.array([np.fromstring(x, dtype=np.float32)
                                      for x in pickle.load(fp)])
        if os.environ['NN_MODULE'] == 'L1':
            nn = lambda x,y: knearest_neighbor_l1.nn(x, y, 1)[0][0]
        #nearest_ind = nearest_neighbor.nn(feat, self.clusters)
        #self.nn = __import__(,
        #                     fromlist=['nn']).nn
        
    def map(self, key, feat):
        feat = np.fromstring(feat, dtype=np.float32)
        nearest_ind = self.nn(feat, self.clusters)
        # Expand the array by 1 and use it to normalize later
        feat = np.resize(feat, feat.shape[0] + 1)
        feat[-1] = 1
        try:
            self.out_sums[nearest_ind] += feat
        except KeyError:
            self.out_sums[nearest_ind] = feat
        yield

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
