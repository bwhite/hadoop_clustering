#!/usr/bin/env python
import os
import time
import itertools
import cPickle as pickle

import numpy as np

import hadoopy

import profile

class Mapper(profile.ProfileJob):
    def __init__(self):
        super(Mapper, self).__init__()
        with open(os.environ["CLUSTERS_PKL"]) as fp:
            self.clusters = pickle.load(fp)
        self.out_sums = {}
        self.nn = __import__(os.environ['NN_MODULE'],
                             fromlist=['nn']).nn

    def map(self, key, feat):
        # Extends the array by 1 dim that has a 1. in it
        feat = np.fromstring(feat + '\x00\x00\x80?', dtype=np.float32)
        nearest_ind = self.nn(feat[0:-1], self.clusters)[0]
        try:
            self.out_sums[nearest_ind] += feat
        except KeyError:
            self.out_sums[nearest_ind] = feat

    def close(self):
        for nearest_ind, out_sum in self.out_sums.iteritems():
            center = out_sum[0:-1] / out_sum[-1]
            yield nearest_ind, center.tostring()
        super(Mapper, self).close()


if __name__ == "__main__":
    if hadoopy.run(Mapper):
        hadoopy.print_doc_quit(__doc__)
