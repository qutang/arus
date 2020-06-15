"""
Computing features about different versions of counts
"""
import numpy as np
from . import transformation
from .. import extensions


def enmo(X):
    """
    Computing ENMO value of accelerometer data

    Arguments:
        X {numpy.ndarray} -- 2D numpy array M x N, M is the number of samples and N is the dimension of the data.

    Returns:
        [numpy.ndarray] -- 1 x 1
    """
    X = extensions.numpy.atleast_float_2d(X)
    vm = transformation.vector_magnitude(X)
    result = np.nanmean(np.clip(vm - 1, a_min=0, a_max=None),
                        axis=0, keepdims=True)
    return result, 'ENMO'
