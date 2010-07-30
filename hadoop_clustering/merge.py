#!/usr/bin/env python
import hadoopy

def mapper(k, v):
    yield k, v

def reducer(k, vs):
    for v in vs:
        yield k, v

if __name__ == "__main__":
    if hadoopy.run(mapper, reducer):
        hadoopy.print_doc_quit(__doc__)
