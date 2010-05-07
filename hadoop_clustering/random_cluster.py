#!/usr/bin/env python
import heapq
import os
import random

import hadoopy


DEFAULT_NUM_CLUSTERS = 100

class Mapper(object):
    def __init__(self):
        self.heap = []
        try:
            self.num_clusters = int(os.environ['NUM_CLUSTERS'])
        except KeyError:
            self.num_clusters = DEFAULT_NUM_CLUSTERS

    def map(self, key, value):
        heapq.heappush(self.heap, (random.random(), value))
        if len(self.heap) >= 2 * self.num_clusters:
            self._compact_heap()

    def _compact_heap(self):
        self.heap = heapq.nsmallest(self.num_clusters, self.heap)

    def close(self):
        self._compact_heap()
        return self.heap


class Reducer(object):
    def __init__(self, out_count=True):
        self.count = 0
        try:
            self.num_clusters = int(os.environ['NUM_CLUSTERS'])
        except KeyError:
            self.num_clusters = DEFAULT_NUM_CLUSTERS
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
    hadoopy.run(Mapper, Reducer, Reducer(False), doc=__doc__)
