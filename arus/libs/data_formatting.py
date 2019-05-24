import numpy as np


def as_scalar_or_np_array(arg):
    if np.isscalar(arg):
        result = np.float64(arg)
    else:
        result = np.array(arg, dtype='float64')
    return result
