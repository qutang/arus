"""
Computing features about different versions of counts
"""
import numpy as np
from .. import extensions as ext

COUNT_FEATURE_NAME_PREFIX = [
    'ENMO'
]


def count_features(X, selected=COUNT_FEATURE_NAME_PREFIX):
    count_funcs = [
        enmo
    ]

    fv = []
    fv_names = []
    for func, prefix in zip(count_funcs, COUNT_FEATURE_NAME_PREFIX):
        if prefix in selected:
            result, names = func(X)
            fv.append(result)
            fv_names += names

    if len(fv) == 0:
        return None, None

    result = np.concatenate(fv, axis=1)
    return result, fv_names


def enmo(X):
    """
    Computing ENMO value of accelerometer data

    Arguments:
        X {numpy.ndarray} -- 2D numpy array M x N, M is the number of samples and N is the dimension of the data.

    Returns:
        [numpy.ndarray] -- 1 x 1
    """
    X = ext.numpy.atleast_float_2d(X)
    vm = ext.numpy.vector_magnitude(X)
    result = np.nanmean(np.clip(vm - 1, a_min=0, a_max=None),
                        axis=0, keepdims=True)
    return result, f'{COUNT_FEATURE_NAME_PREFIX[0]}_0'
