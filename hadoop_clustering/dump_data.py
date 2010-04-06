#!/usr/bin/env python
import sys
import os
import cPickle as pickle

import numpy as np

import hadoopy
from hadoopy.pickle import b64dec, b64enc
import simplejson as json


class Mapper(object):
    def __init__(self, io_method):
        self.in_func = {'b64': self.b64, 'json': self.json}[io_method]

    def b64(self, value):
        return np.fromstring(b64dec(value), dtype=np.float32)

    def json(self, value):
        return np.array(json.loads(value), dtype=np.float32)
    
    def map(self, key, value):
        yield json.dumps(self.in_func(value).tolist())


if __name__ == "__main__":
    try:
        io_method = os.environ["IO_METHOD"]
    except KeyError:
        hadoopy.print_doc_quit(__doc__)
    if hadoopy.run(Mapper(io_method)):
        hadoopy.print_doc_quit(__doc__)
