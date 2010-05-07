#!/usr/bin/env python
import os
import numpy as np
import hadoopy


class Mapper(object):
    def __init__(self):
        self.num_clusters = int(os.environ["NUM_CLUSTERS"])
        self.num_points = int(os.environ["NUM_POINTS"])
        self.num_dims = int(os.environ["NUM_DIMS"])
        self.dead = False

    def map(self, key, value):
        if not self.dead:
            self.dead = True
            for i in range(self.num_clusters):
                rnd = np.random.random((self.num_points, self.num_dims)) + i
                for j in np.array(rnd, dtype=np.float32):
                    yield i, j.tostring()

def reducer(key, values):
    for value in values:
        yield key, value

if __name__ == "__main__":
    hadoopy.run(Mapper, reducer, doc=__doc__)
