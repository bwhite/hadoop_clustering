#!/usr/bin/env python
import os
import random

import numpy as np

import hadoopy


class MapReduce(object):
    def __init__(self):
        self.canopies = np.array([])
        try:
            nn_mod = os.environ['NN_MODULE']
        except KeyError:
            nn_mod = 'nn_l1'

        self.nn = __import__(nn_mod,
                             fromlist=['nn']).nn
        try:
            self.soft_dist = os.environ['CANOPY_SOFT_DIST']
            self.hard_dist = os.environ['CANOPY_HARD_DIST']
        except KeyError:
            self.soft_dist = 1
            self.hard_dist = .25
        
    def map(self, key, feat):
        feat = np.array([np.fromstring(feat, dtype=np.float32)])
        if self.canopies.size:
            nearest_dist = self.nn(feat, self.canopies)[1]
            if nearest_dist > self.hard_dist:
                self.canopies = np.concatenate((self.canopies, feat))
        else:
            self.canopies = feat

    def reduce(self, key, feats):
        for feat in feats:
            self.map(key, feat)

    def _random_canopy(self, canopies):
        return np.array(random.sample(canopies, 1))

    def close(self):
        final_canopies = self._random_canopy(self.canopies)
        uncovered_points = True
        while uncovered_points:
            uncovered_points = False
            valid_canopies = []
            for x in self.canopies:
                nearest_dist = self.nn(x, final_canopies)[1]
                if nearest_dist > self.soft_dist:
                    uncovered_points = True
                if nearest_dist > self.hard_dist:
                    valid_canopies.append(x)
            if uncovered_points:
                canopy = self._random_canopy(valid_canopies)
                final_canopies = np.concatenate((final_canopies, canopy))
                self.canopies = valid_canopies
        for canopy in final_canopies:
            yield random.random(), canopy.tostring()


if __name__ == "__main__":
    if hadoopy.run(MapReduce, MapReduce):
        hadoopy.print_doc_quit(__doc__)
