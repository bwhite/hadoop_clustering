#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np

import hadoopy
from kmeans_cluster import reducer

# TODOS
# Canopy Clust TODOs
# Make a script that takes in points and generates the input for this (nearest canopies)

class Mapper(object):
    def __init__(self):
        self.out_sums = {}
        with open(os.environ["CLUSTERS_PKL"]) as fp:
            self.clusters = pickle.load(fp)
        with open(os.environ["CANOPIES_PKL"]) as fp:
            self.canopies = pickle.load(fp)
        try:
            self.soft_dist = float(os.environ['CANOPY_SOFT_DIST'])
        except KeyError:
            self.soft_dist = 1.
        # TODO Find the clusters that are within the soft dist to the canopy
        self.canopy_clusters = []
        for canopy in self.canopies:
            dists = [np.linalg.norm(x - canopy, 1) for x in self.clusters]
            clust = set([x for x, y in enumerate(dists) if y < self.soft_dist])
            self.canopy_clusters.append(clust)
        self.nn = __import__(os.environ['NN_MODULE'],
                             fromlist=['nn']).nn
        
    def map(self, key, feat):
        feat = np.fromstring(feat, dtype=np.float32)
        dists = (np.linalg.norm(x - canopy, 1) for x in self.canopies)
        canopies = set([x for x, y in enumerate(dists) if y < self.soft_dist])
        cluster_ids = sum([self.canopy_clusters[x] for x in canopies], [])
        clusters = np.array([self.clusters[x] for x in set(cluster_ids)])
        # Find NN using slow metric
        nearest_ind = self.nn(feat, clusters)
        nearest_ind = cluster_ids[nearest_ind]
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


if __name__ == "__main__":
    if hadoopy.run(Mapper, reducer):
        hadoopy.print_doc_quit(__doc__)
