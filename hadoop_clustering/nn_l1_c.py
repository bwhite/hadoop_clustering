import numpy as np
from vitrieve_algorithms import knearest_neighbor_l1


def nn(feature, clusters):
    """Find L1 nearest neighbor

    >>> nn(np.array([1.]), np.array([[2.], [1.]]))
    1

    Args:
        feature: A numpy array of shape (N,) or (1, N). (N=Dims)
        clusters: A numpy array of shape (M, N). (N=Dims, M=NumClusters)
    Returns:
        An int representing the nearest neighbor index into clusters.
    """
    feature = np.array(feature, dtype=np.float32)
    clusters = np.array(clusters, dtype=np.float32)
    return knearest_neighbor_l1.nn(feature, clusters, 1)[0][0]
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
