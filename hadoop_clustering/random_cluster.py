#!/usr/bin/env python
import os
import random

import hadoopy


def mapper(key, value):
    yield random.random(), value


class Reducer(object):
    def __init__(self, out_count=True):
        self.count = 0
        try:
            self.num_clusters = int(os.environ['NUM_CLUSTERS'])
        except KeyError:
            self.num_clusters = 100
        self.output = self.yield_count if out_count else self.yield_key

    def yield_count(self, key, value):
        return self.count, value

    def yield_key(self, key, value):
        return key, value
        
    def reduce(self, key, values):
        for value in values:
            if self.count < self.num_clusters:
                yield self.output(key, value)
                self.count += 1


if __name__ == "__main__":
    hadoopy.run(mapper, Reducer, Reducer(False), doc=__doc__)
