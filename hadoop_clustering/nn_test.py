import unittest
import numpy as np
import timeit

class TestNN(unittest.TestCase, object):
    
    def __init__(self, name='runTest'):
        super(TestNN, self).__init__(name)

    def setUp(self):
        from nn_l1 import nn
        self.nn = nn
        self.nn_baseline = nn

    def test_nn(self):
        self.assertEqual(self.nn(np.array([1]), np.array([[1], [2]])), 0)
        self.assertEqual(self.nn(np.array([2]), np.array([[1], [2]])), 1)
        self.assertEqual(self.nn(np.array([1]), np.array([[2], [1]])), 1)
        self.assertEqual(self.nn(np.array([2]), np.array([[2], [1]])), 0)

    def _init_samples(self, dims, clusters):
        self.f = np.array(np.random.random(size=(dims,)), dtype=np.float32)
        self.c = np.array(np.random.random(size=(clusters, dims)), dtype=np.float32)

    def test_speed(self, dims=4096, clusters=100, rounds=100):
        ti = timeit.Timer(stmt= lambda: self.nn(self.f, self.c),
                          setup=lambda: self._init_samples(dims, clusters))
        print('Time[%f]' % (ti.timeit(rounds)))

    def test_fuzz(self, dims=5, clusters=5, rounds=10000):
        for i in range(rounds):
            self._init_samples(dims, clusters)
            self.assertEqual(self.nn(self.f, self.c), self.nn_baseline(self.f, self.c))

if __name__ == '__main__':
    unittest.main()
