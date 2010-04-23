#!/usr/bin/env python                                                                                                                                                 
import unittest
import hadoopy
import numpy as np
from canopy_cluster import MapReduce


class Test(hadoopy.Test):

    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_map_badimg(self):
        test_in = [('blah', np.array(np.random.random(1) * 5, dtype=np.float32).tostring()) for x in range(10000)]
        out = self.call_map(MapReduce, test_in)
        print('output:[%s][%d]' % (str(out), len(out)))
        out = sum(sorted([np.fromstring(x[1], dtype=np.float32).tolist() for x in out]), [])
        print(out)
            

if __name__ == '__main__':
    unittest.main()
