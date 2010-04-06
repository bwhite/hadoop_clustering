#!/usr/bin/env python
import sys
import os
import cPickle as pickle

import numpy as np
import simplejson as json

import hadoopy
from hadoopy.pickle import b64dec, b64enc


class Mapper(object):
    def __init__(self, num_points, num_dims):
        self.num_points = int(os.environ["NUM_POINTS"])
        self.num_dims = int(os.environ["NUM_DIMS"])
        self.dead = False

    def map(self, key, value):
        if not self.dead:
            self.dead = True
            for i in range(num_points):
                yield i, np.random(self.num_dims).tostring()

if __name__ == "__main__":
    hadoopy.run(Mapper, doc=__doc__)
