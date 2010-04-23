#!/usr/bin/env python                                                                                                                                                 
import unittest
import hadoopy
import numpy as np
import os
from canopy_cluster import MapReduce

from IPython.Shell import IPShellEmbed

def plot_clustering(points, clusters):
    import matplotlib.pyplot as mp
    points = [np.fromstring(x[1], dtype=np.float32).tolist() for x in points]
    clusters = [np.fromstring(x[1], dtype=np.float32).tolist() for x in clusters]
    mp.scatter(*zip(*points), c='b')
    mp.scatter(*zip(*clusters), c='r')
    mp.show()


class Test(hadoopy.Test):

    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_map_badimg(self):
        os.environ['CANOPY_SOFT_DIST'] = str(1.)
        os.environ['CANOPY_HARD_DIST'] = str(.25)
        test_in = [('blah', np.array(np.random.random(2) * 5, dtype=np.float32).tostring()) for x in range(10000)]
        out = self.call_map(MapReduce, test_in)
        print('output:[%s][%d]' % (str(out), len(out)))
        print(out)
        plot_clustering(test_in, out)
        out = self.call_reduce(MapReduce, test_in)

        #IPShellEmbed()()
        
            

if __name__ == '__main__':
    unittest.main()
