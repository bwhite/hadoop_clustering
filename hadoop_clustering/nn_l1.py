import numpy as np


def nn(feature, clusters):
    """Find L1 nearest neighbor

    >>> nn(np.array([1]), np.array([[1], [2]]))
    (0, 0.0)
    >>> nn(np.array([1.]), np.array([[2.], [1.]]))
    (1, 0.0)
    >>> nn(np.array([2]), np.array([[1], [2]]))
    (1, 0.0)
    >>> nn(np.array([[1]]), np.array([[1], [2]]))
    (0, 0.0)

    Args:
        feature: A numpy array of shape (N,) or (1, N). (N=Dims)
        clusters: A numpy array of shape (M, N). (N=Dims, M=NumClusters)
    Returns:
        An int representing the nearest neighbor index into clusters.
    """
    dist = np.sum(np.abs(feature-clusters), 1)
    ind = int(np.argmin(dist))
    return ind, float(dist[ind])
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
