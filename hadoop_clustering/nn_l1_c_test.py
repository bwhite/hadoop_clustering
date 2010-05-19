import unittest
import nn_l1_test

class TestNNC(nn_l1_test.TestNN):
    def __init__(self, name='runTest'):
        super(TestNNC, self).__init__(name)

    def setUp(self):
        super(TestNNC, self).setUp()
        from nn_l1_c import nn
        self.nn = nn

if __name__ == '__main__':
    unittest.main()
