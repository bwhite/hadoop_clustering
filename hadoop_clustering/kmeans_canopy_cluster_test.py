#!/usr/bin/env python                                                                                                                                                 
import unittest
import hadoopy
import numpy as np
import os
from kmeans_canopy_cluster import Mapper

from IPython.Shell import IPShellEmbed

# Data dump
a = [((0, [10, 21, 36, 148, 152, 156]), '\x96jr\xbfqe\x85='),
     ((0, [10, 21, 36, 134, 148, 152]), 'E\x92A>I33\xbf'),
     ((0, [10, 21, 36, 148, 152, 156]), 'N\x07\xed\xbe~\xe6@='),
     ((0, [10, 21, 36, 134, 148, 152, 156]), '<\x7f[>/\xd5\x84\xbd'),
     ((0, [10, 21, 36, 148, 152, 156]), '\x17+\x13\xbfGX5<'),
     ((0, [10, 134, 148, 152]), '\xdd3\x81?+c\xe0='),
     ((0, [10, 21, 36, 134, 148, 152]), '\x1b\xd4\xe2>\xdbnq\xbf'),
     ((0, [10, 21, 36, 134, 152]), 'QP\x1b?\xbe\xdap\xbf'),
     ((0, [10, 21, 36, 134, 148, 152]), '\xa3\x03\xad\xbd\x0f\x8e0\xbf'),
     ((0, [10, 21, 36, 152]), '\xfc\xa7\xda\xbe|\x02\xce\xbf'),
     ((0, [21, 36, 148, 152, 156]), 'e\xcc\xc3\xbf\xbd\xae)>'),
     ((0, [10, 21, 36, 134, 148, 152, 156]), '!\xec\xd5=\r\xf2\xbb='),
     ((0, [107, 134, 148, 152, 153, 171]), 'i\x14\x99?Q\x04\x94?'),
     ((0, [36, 134, 148, 152, 156]), '\xcdFI\xbe;\x00\x02?'),
     ((0, [134, 148, 152]), '\x0bAy?\x0e\xa3\x8a>'),
     ((0, [10, 110, 134, 152]), '\xc1H\xc6?\xad\xbck\xbf'),
     ((0, [10, 21, 36, 148, 152, 156]), '\\\xbd\x00\xbfm\xcb\x99\xbe'),
     ((0, [21, 36, 148, 152, 156]), '\x08Y$\xbf\xa39\x18?'),
     ((0, [10, 36, 134, 148, 152, 156]), ';i6>\xb4U\x89>'),
     ((0, [10, 21, 36, 148, 152, 156]), '\x92,\x02\xbf\x99\x8f\x94\xbe')]


class Test(hadoopy.Test):

    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_map_badimg(self):
        os.environ['CANOPY_SOFT_DIST'] = str(8.)
        os.environ['CLUSTERS_PKL'] = 'clusters.pkl'
        os.environ['NN_MODULE'] = 'nn_l2sqr'
        out = self.call_map(Mapper, a)

if __name__ == '__main__':
    unittest.main()
