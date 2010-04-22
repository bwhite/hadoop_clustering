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
            var = np.eye(self.num_dims)
            for i in range(num_clusters):
                mu = [i] * self.num_dims
                rnd = np.random.multivariate_normal(mu, var, num_points)
                for j in np.array(rnd, dtype=np.float32):
                    yield i, j.tostring()

if __name__ == "__main__":
    hadoopy.run(Mapper, doc=__doc__)
