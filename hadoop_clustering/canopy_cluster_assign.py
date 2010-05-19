#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np

import hadoopy

class Mapper(object):
    def __init__(self):
        with open(os.environ["CANOPIES_PKL"]) as fp:
            self.canopies = pickle.load(fp)
        try:
            self.soft_dist = float(os.environ['CANOPY_SOFT_DIST'])
        except KeyError:
            self.soft_dist = 1.

    @staticmethod
    def _l2sqr_thresh(point, clusters, thresh):
        dist = point - clusters
        dist *= dist
        dist = np.sum(dist, 1)
        np.less_equal(dist, thresh, dist)
        return np.nonzero(dist)[0].tolist()

    def map(self, key, feat):
        feat_arr = np.fromstring(feat, dtype=np.float32)
        canopies = self._l2sqr_thresh(feat_arr, self.canopies, self.soft_dist) # TODO Generalize
        yield (key, canopies), feat

if __name__ == "__main__":
    if hadoopy.run(Mapper):
        hadoopy.print_doc_quit(__doc__)
