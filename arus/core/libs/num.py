import numpy as np


def atleast_scalar(arg):
    if np.isscalar(arg):
        result = np.float64(arg)
    else:
        result = np.array(arg, dtype='float64')
    return result


def format_arr(arr):
    arr = np.atleast_2d(arr)
    arr = np.float64(arr)
    return arr
